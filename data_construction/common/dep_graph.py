#!/usr/bin/env python3
"""
dep_graph.py — Dependency-Aware Fill-in-the-Middle function selection.

The scoring core shared by both pipelines. Given Python source files (dicts with
a 'code_content' field) it builds a per-file program dependency graph and scores
every function on two axes:

    Ĥ(v)  intrinsic complexity   — LOC, cyclomatic complexity, AST depth
    Î(v)  contextual inferability — what the *rest of the file* tells you about v
                                    (call sites, callees, signature, docstring,
                                     sibling methods)

    FIM_Score(v) = Ĥ(v) · Î(v) / (Ĥ(v) + Î(v))

peaking where a function is both substantial and recoverable from its context,
with a one-sided Gaussian penalty for functions whose difficulty exceeds what
the context can support. This is the library; the CLIs live in
single_function/step_3_select_functions.py (individual functions) and
multi_function/step_3_select_function_groups.py (coupled groups of 2-3).
"""

import ast
import math
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any

try:
    from tqdm import tqdm
except ImportError:  # tqdm is a convenience, not a requirement
    def tqdm(x, **kwargs):
        return x


# ================================================================
# Data Structures
# ================================================================

@dataclass
class FunctionInfo:
    """Metadata and computed metrics for a single function/method."""
    name: str                    # Qualified: "ClassName.method" or "func"
    lineno: int                  # Start line (1-indexed, line of `def`)
    end_lineno: int              # End line (1-indexed, inclusive)
    body_lineno: int             # First line of function body (1-indexed)
    source_text: str             # Raw source code
    loc: int                     # Lines of code
    params: List[str]            # Parameter names (including self/cls)
    has_return_type: bool
    has_param_types: bool
    has_docstring: bool
    class_name: Optional[str]    # Owning class, None for module-level
    decorators: List[str] = field(default_factory=list)
    calls: Set[str] = field(default_factory=set)
    called_by: Set[str] = field(default_factory=set)
    cyclomatic_complexity: int = 1
    max_ast_depth: int = 0
    num_variables: int = 0


@dataclass
class FIMCandidate:
    """A function selected as a FIM mask target, with all scores."""
    func_name: str
    start_line: int
    end_line: int
    body_lineno: int
    source_text: str
    loc: int
    complexity: float
    inferability: float
    fim_score: float
    difficulty: float


# ================================================================
# Step 1 — Dependency Graph Builder
# ================================================================

class DependencyGraphBuilder:
    """
    Parse a Python source file and construct a dependency graph.

    Nodes  = functions / methods
    Edges  = call edges  | sibling edges (same-class methods)
    """

    def __init__(self, source: str):
        self.source = source
        self.source_lines = source.splitlines()
        self.tree: Optional[ast.Module] = None
        self.functions: Dict[str, FunctionInfo] = {}
        self.ast_nodes: Dict[str, ast.AST] = {}       # name -> AST node
        self.edges: List[Tuple[str, str, str]] = []    # (src, dst, type)
        self._short_to_qname: Dict[str, List[str]] = defaultdict(list)

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def build(self) -> bool:
        """Parse source and build graph.  Returns True on success."""
        try:
            self.tree = ast.parse(self.source)
        except Exception as e:
            print(e)
            return False

        # Pass 1 — collect every function / method definition
        self._collect_functions(self.tree, current_class=None)

        # Short-name index for fuzzy call resolution
        for fname in self.functions:
            short = fname.split(".")[-1]
            self._short_to_qname[short].append(fname)

        # Pass 2 — extract calls inside each function body
        for fname, finfo in self.functions.items():
            self._analyze_calls(fname, finfo)

        # Pass 3 — create call edges (exact + fuzzy matching)
        self._build_call_edges()

        # Pass 4 — sibling edges (same class)
        self._add_sibling_edges()

        # Pass 5 — compute complexity metrics
        for fname in self.functions:
            self._compute_metrics(fname)

        return True

    # ----------------------------------------------------------------
    # Function collection
    # ----------------------------------------------------------------

    def _collect_functions(self, node: ast.AST, current_class: Optional[str]):
        """Walk top-level and class-level statements to find func defs."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                self._collect_functions(child, current_class=child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._register_function(child, current_class)

    def _register_function(self, node, current_class: Optional[str]):
        qname = f"{current_class}.{node.name}" if current_class else node.name

        # Source text extraction
        start_idx = node.lineno - 1                       # 0-indexed
        end_line = getattr(node, "end_lineno", None)
        if end_line is None:
            end_line = self._fallback_end(node)            # Python < 3.8
        source_text = "\n".join(self.source_lines[start_idx:end_line])
        loc = end_line - start_idx

        # Body start line
        if node.body:
            body_lineno = node.body[0].lineno
        else:
            body_lineno = node.lineno + 1  # fallback for empty body

        # Collect all argument objects
        all_args = (
            node.args.args
            + getattr(node.args, "posonlyargs", [])
            + node.args.kwonlyargs
        )

        # Docstring check
        body0 = node.body[0] if node.body else None
        has_docstring = (
            isinstance(body0, ast.Expr)
            and isinstance(getattr(body0, "value", None), ast.Constant)
            and isinstance(body0.value.value, str)
        )
        # Fallback for Python 3.7 ast.Str
        if not has_docstring and hasattr(ast, "Str"):
            has_docstring = (
                isinstance(body0, ast.Expr)
                and isinstance(body0.value, ast.Str)
            )

        # Decorators
        decorators = []
        for d in node.decorator_list:
            if isinstance(d, ast.Name):
                decorators.append(d.id)
            elif isinstance(d, ast.Attribute):
                decorators.append(d.attr)
            elif isinstance(d, ast.Call) and isinstance(d.func, ast.Name):
                decorators.append(d.func.id)

        self.functions[qname] = FunctionInfo(
            name=qname,
            lineno=node.lineno,
            end_lineno=end_line,
            body_lineno=body_lineno,
            source_text=source_text,
            loc=loc,
            params=[a.arg for a in node.args.args],
            has_return_type=(node.returns is not None),
            has_param_types=any(a.annotation is not None for a in all_args),
            has_docstring=has_docstring,
            class_name=current_class,
            decorators=decorators,
        )
        self.ast_nodes[qname] = node

    def _fallback_end(self, node) -> int:
        """Estimate end line for Python < 3.8 (no end_lineno)."""
        m = node.lineno
        for child in ast.walk(node):
            if hasattr(child, "lineno"):
                m = max(m, child.lineno)
        return m

    # ----------------------------------------------------------------
    # Call analysis
    # ----------------------------------------------------------------

    def _analyze_calls(self, fname: str, finfo: FunctionInfo):
        """Walk function body AST and record every call target."""
        node = self.ast_nodes.get(fname)
        if node is None:
            return
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                resolved = self._resolve_call(child, finfo.class_name)
                if resolved:
                    finfo.calls.add(resolved)

    def _resolve_call(
        self, call: ast.Call, current_class: Optional[str]
    ) -> Optional[str]:
        func = call.func

        if isinstance(func, ast.Name):
            name = func.id
            # Class instantiation  → __init__
            init_qname = f"{name}.__init__"
            if init_qname in self.functions:
                return init_qname
            # Direct module-level function
            if name in self.functions:
                return name
            return name  # keep for fuzzy matching later

        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name):
                obj, attr = func.value.id, func.attr
                # self.method() / cls.method()
                if obj in ("self", "cls") and current_class:
                    return f"{current_class}.{attr}"
                # ClassName.static_method()
                candidate = f"{obj}.{attr}"
                if candidate in self.functions:
                    return candidate
            # Fallback: just the attribute name
            return func.attr

        return None

    def _build_call_edges(self):
        """Create directed call edges, using fuzzy short-name matching."""
        for fname, finfo in self.functions.items():
            for callee in finfo.calls:
                targets: Set[str] = set()
                if callee in self.functions:
                    targets.add(callee)
                else:
                    short = callee.split(".")[-1]
                    for qn in self._short_to_qname.get(short, []):
                        if qn != fname:
                            targets.add(qn)
                for t in targets:
                    self.edges.append((fname, t, "call"))
                    self.functions[t].called_by.add(fname)

    def _add_sibling_edges(self):
        """Undirected edges between methods of the same class."""
        cls_map: Dict[str, List[str]] = defaultdict(list)
        for fn, fi in self.functions.items():
            if fi.class_name:
                cls_map[fi.class_name].append(fn)
        for methods in cls_map.values():
            for i, a in enumerate(methods):
                for b in methods[i + 1 :]:
                    self.edges.append((a, b, "sibling"))
                    self.edges.append((b, a, "sibling"))

    # ----------------------------------------------------------------
    # Complexity metrics
    # ----------------------------------------------------------------

    def _compute_metrics(self, fname: str):
        fi = self.functions[fname]
        node = self.ast_nodes.get(fname)
        if node is None:
            return
        fi.cyclomatic_complexity = self._cyclomatic_complexity(node)
        fi.max_ast_depth = self._max_ast_depth(node)
        fi.num_variables = self._count_variables(node)

    @staticmethod
    def _cyclomatic_complexity(node: ast.AST) -> int:
        cc = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.IfExp)):
                cc += 1
            elif isinstance(child, (ast.For, ast.AsyncFor, ast.While)):
                cc += 1
            elif isinstance(child, ast.ExceptHandler):
                cc += 1
            elif isinstance(child, ast.BoolOp):
                cc += len(child.values) - 1
            elif isinstance(
                child,
                (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
            ):
                cc += len(child.generators)
        return cc

    @staticmethod
    def _max_ast_depth(node: ast.AST, depth: int = 0) -> int:
        max_d = depth
        nesting = (
            ast.If, ast.While, ast.For, ast.AsyncFor,
            ast.With, ast.AsyncWith, ast.Try,
        )
        if hasattr(ast, "TryStar"):
            nesting = nesting + (ast.TryStar,)
        for child in ast.iter_child_nodes(node):
            nd = depth + 1 if isinstance(child, nesting) else depth
            max_d = max(max_d, DependencyGraphBuilder._max_ast_depth(child, nd))
        return max_d

    @staticmethod
    def _count_variables(node: ast.AST) -> int:
        names: Set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for t in child.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
                    elif isinstance(t, (ast.Tuple, ast.List)):
                        names.update(
                            e.id for e in t.elts if isinstance(e, ast.Name)
                        )
            elif isinstance(child, ast.AugAssign):
                if isinstance(child.target, ast.Name):
                    names.add(child.target.id)
            elif isinstance(child, ast.AnnAssign):
                if isinstance(getattr(child, "target", None), ast.Name):
                    names.add(child.target.id)
        return len(names)


# ================================================================
# Steps 2–4 — FIM Scoring & Threshold Selection
# ================================================================

class FIMSelector:
    """
    Compute information-theoretic FIM scores and select mask targets.

    Uses **threshold-based** filtering — a file with no suitable
    functions produces an empty target list.
    """

    DEFAULT_CONFIG = {
        # --- File-level bounds ---
        "min_file_lines": 50,
        "max_file_lines": 1800,
        # --- Function LOC bounds ---
        "min_loc": 10,
        "max_loc": 200,
        # --- Complexity weights  (Ĥ) ---
        "w_loc": 0.4,
        "w_cc": 0.4,
        "w_depth": 0.2,
        # --- Inferability weights (Î) ---
        "alpha_caller": 0.30,
        "beta_callee": 0.25,
        "gamma_sig": 0.20,
        "delta_doc": 0.10,
        "epsilon_class": 0.15,
        # --- Difficulty penalty (one-sided) ---
        "difficulty_ceiling": 0.5,
        "difficulty_sigma": 0.20,
        # --- Hard thresholds ---
        "score_threshold": 0.08,
        "min_complexity": 0.15,
    }

    # Dunder methods: usually required as context, not mask targets
    SKIP_NAMES = frozenset({
        "__init__", "__new__", "__del__", "__repr__", "__str__",
        "__len__", "__getitem__", "__setitem__", "__delitem__",
        "__iter__", "__next__", "__contains__",
        "__enter__", "__exit__", "__hash__",
        "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
        "__bool__", "__call__", "__format__", "__sizeof__",
    })

    def __init__(
        self,
        functions: Dict[str, FunctionInfo],
        edges: List[Tuple[str, str, str]],
        ast_nodes: Dict[str, ast.AST],
        config: Optional[Dict] = None,
    ):
        self.functions = functions
        self.ast_nodes = ast_nodes
        self.cfg = {**self.DEFAULT_CONFIG, **(config or {})}

        # Build adjacency indexes
        self.callers: Dict[str, Set[str]] = defaultdict(set)
        self.callees: Dict[str, Set[str]] = defaultdict(set)
        self.siblings: Dict[str, Set[str]] = defaultdict(set)
        for src, dst, etype in edges:
            if etype == "call":
                self.callers[dst].add(src)
                self.callees[src].add(dst)
            elif etype == "sibling":
                self.siblings[src].add(dst)

    # ----------------------------------------------------------------
    # Ĥ(v) — intrinsic complexity proxy
    # ----------------------------------------------------------------

    def compute_complexity(self, fi: FunctionInfo) -> float:
        c = self.cfg
        norm_loc = min(fi.loc / 50.0, 2.0)
        norm_cc = min(fi.cyclomatic_complexity / 10.0, 2.0)
        norm_depth = min(fi.max_ast_depth / 5.0, 2.0)
        return (
            c["w_loc"] * norm_loc
            + c["w_cc"] * norm_cc
            + c["w_depth"] * norm_depth
        )

    # ----------------------------------------------------------------
    # Î(v) — contextual inferability proxy
    # ----------------------------------------------------------------

    def compute_inferability(self, fname: str) -> float:
        c = self.cfg
        fi = self.functions[fname]

        # 1) Caller information
        caller_sp = 0.0
        for caller in self.callers.get(fname, set()):
            if caller in self.functions:
                caller_sp += self._call_site_specificity(caller, fname)
        caller_score = min(caller_sp / 3.0, 1.0)

        # 2) Callee information (file-internal only)
        internal_callees = [
            x for x in self.callees.get(fname, set()) if x in self.functions
        ]
        callee_score = min(len(internal_callees) / 4.0, 1.0)

        # 3) Signature information
        sig_score = 0.0
        if fi.has_return_type:
            sig_score += 0.30
        if fi.has_param_types:
            sig_score += 0.25
        raw_name = fname.split(".")[-1]
        if not raw_name.startswith("_"):
            parts = [p for p in raw_name.split("_") if p]
            sig_score += min(len(parts) / 5.0, 0.25)
        non_self_params = [p for p in fi.params if p not in ("self", "cls")]
        sig_score += min(len(non_self_params) / 6.0, 0.20)
        sig_score = min(sig_score, 1.0)

        # 4) Documentation
        doc_score = 0.5 if fi.has_docstring else 0.0

        # 5) Class context
        class_score = 0.0
        if fi.class_name:
            n_sibs = len(self.siblings.get(fname, set()))
            class_score = min(n_sibs / 5.0, 1.0)
            init_name = f"{fi.class_name}.__init__"
            if init_name in self.functions and init_name != fname:
                class_score = min(class_score + 0.3, 1.0)

        return (
            c["alpha_caller"] * caller_score
            + c["beta_callee"] * callee_score
            + c["gamma_sig"] * sig_score
            + c["delta_doc"] * doc_score
            + c["epsilon_class"] * class_score
        )

    def _call_site_specificity(self, caller: str, callee: str) -> float:
        """
        Estimate information provided by the call site(s) in `caller`
        that invoke `callee`.  Considers argument types and counts.
        """
        node = self.ast_nodes.get(caller)
        if node is None:
            return 0.5

        short = callee.split(".")[-1]
        total_sp = 0.0
        found = False

        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            n = None
            if isinstance(child.func, ast.Name):
                n = child.func.id
            elif isinstance(child.func, ast.Attribute):
                n = child.func.attr
            if n != short:
                continue

            found = True
            sp = 0.50
            for arg in child.args:
                if isinstance(arg, ast.Constant):
                    sp += 0.15
                elif isinstance(arg, ast.Name):
                    sp += 0.05
                else:
                    sp += 0.08
            sp += 0.12 * len(child.keywords)
            sp += 0.10
            total_sp += sp

        return min(total_sp, 1.5) if found else 0.5

    # ----------------------------------------------------------------
    # FIM score & selection
    # ----------------------------------------------------------------

    def select_targets(self) -> List[FIMCandidate]:
        """
        Score every function and return those above the threshold.

        FIM_Score = Ĥ(v) × Î(v) / (Ĥ(v) + Î(v))
        penalised by a one-sided Gaussian when difficulty > ceiling.
        """
        c = self.cfg
        results: List[FIMCandidate] = []

        for fname, fi in self.functions.items():
            # ---- hard filters ----
            if fi.loc < c["min_loc"] or fi.loc > c["max_loc"]:
                continue
            if fname.split(".")[-1] in self.SKIP_NAMES:
                continue

            h = self.compute_complexity(fi)
            if h < c["min_complexity"]:
                continue

            i_val = self.compute_inferability(fname)

            # ---- information-theoretic score ----
            raw_score = h * (i_val / (h + i_val + 1e-8))

            # Difficulty: fraction of complexity unexplained by context
            difficulty = max(0.0, h - i_val) / (h + 1e-8)

            # One-sided Gaussian penalty for too-hard functions
            if difficulty > c["difficulty_ceiling"]:
                excess = difficulty - c["difficulty_ceiling"]
                factor = math.exp(
                    -(excess ** 2) / (2.0 * c["difficulty_sigma"] ** 2)
                )
            else:
                factor = 1.0

            final_score = raw_score * factor

            # ---- threshold filter ----
            if final_score < c["score_threshold"]:
                continue

            results.append(
                FIMCandidate(
                    func_name=fname,
                    start_line=fi.lineno,
                    end_line=fi.end_lineno,
                    body_lineno=fi.body_lineno,
                    source_text=fi.source_text,
                    loc=fi.loc,
                    complexity=round(h, 4),
                    inferability=round(i_val, 4),
                    fim_score=round(final_score, 4),
                    difficulty=round(difficulty, 4),
                )
            )

        results.sort(key=lambda x: x.fim_score, reverse=True)
        return results


# ================================================================
# Masking Utility
# ================================================================

def mask_function_body(
    code_content: str,
    start_line: int,
    end_line: int,
    body_lineno: int,
) -> str:
    """
    Replace the body of a function with ``# <MASKED_FUNCTION_BODY>``.

    Parameters
    ----------
    code_content : str
        Full file source code.
    start_line : int
        1-indexed line of the ``def`` keyword.
    end_line : int
        1-indexed last line of the function (inclusive).
    body_lineno : int
        1-indexed first line of the function body.

    Returns
    -------
    str
        The code with the target function's body replaced by a single
        comment line ``# <MASKED_FUNCTION_BODY>`` at the correct
        indentation.
    """
    lines = code_content.splitlines()
    body_idx = body_lineno - 1  # 0-indexed start of body

    # Determine body indentation from the first body line
    if body_idx < len(lines):
        body_line = lines[body_idx]
        indent = len(body_line) - len(body_line.lstrip())
    else:
        # Fallback: def-line indent + 4 spaces
        def_idx = start_line - 1
        if def_idx < len(lines):
            indent = len(lines[def_idx]) - len(lines[def_idx].lstrip()) + 4
        else:
            indent = 4

    mask_line = " " * indent + "# <MASKED_FUNCTION_BODY>"

    # lines[:body_idx]  → everything up to (not including) body start
    # [mask_line]        → the single mask placeholder
    # lines[end_line:]   → everything after the function
    #   (end_line is 1-indexed inclusive, so 0-indexed exclusive = end_line)
    new_lines = lines[:body_idx] + [mask_line] + lines[end_line:]
    return "\n".join(new_lines)


# ================================================================
# Post-processing
# ================================================================

def postprocess_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Expand sample-level results into **function-level** entries.

    Each mask target becomes its own dict containing:
      - all original sample fields (except mask_targets / mask_target_count / skip_reason / code_content)
      - all function-level scoring fields
      - ``masked_code_content``: the file source with that function body masked

    Validation: for each target, verify that the source_text extracted
    from code_content by [start_line, end_line] matches the stored
    source_text.  Mismatches are logged and skipped.

    The original ``code_content`` is dropped from each entry to save space.
    The returned list is sorted by ``fim_score`` descending.
    """
    function_entries: List[Dict[str, Any]] = []
    total_targets = 0
    skipped_targets = 0

    for sample in results:
        targets = sample.get("mask_targets", [])
        if not targets:
            continue

        code_content = sample.get("code_content", "")
        code_lines = code_content.splitlines()

        # Sample-level fields to carry forward (drop heavy / internal keys)
        sample_fields = {
            k: v
            for k, v in sample.items()
            if k not in (
                "mask_targets", "mask_target_count",
                "skip_reason", "code_content",
            )
        }

        for target in targets:
            total_targets += 1

            # --- Validation: re-extract source from code_content and compare ---
            start = target["start_line"]   # 1-indexed
            end = target["end_line"]       # 1-indexed inclusive
            expected_text = target.get("func_content", "")

            extracted_text = "\n".join(code_lines[start - 1 : end])

            if extracted_text != expected_text:
                # Try stripping trailing whitespace per line as a lenient check
                extracted_stripped = "\n".join(
                    l.rstrip() for l in code_lines[start - 1 : end]
                )
                expected_stripped = "\n".join(
                    l.rstrip() for l in expected_text.splitlines()
                )
                if extracted_stripped != expected_stripped:
                    skipped_targets += 1
                    sid = sample.get("sample_id", "?")
                    fname = target.get("func_name", "?")
                    print(
                        f"  [WARN] sample {sid}, func {fname}: "
                        f"source_text mismatch "
                        f"(extracted {len(extracted_text)} chars vs "
                        f"stored {len(expected_text)} chars) → skipped"
                    )
                    continue

            entry = dict(sample_fields)          # shallow copy of sample info
            entry.update(target)                 # overlay function-level fields

            # Build the masked version of the full file
            entry["masked_code_content"] = mask_function_body(
                code_content,
                target["start_line"],
                target["end_line"],
                target["body_lineno"],
            )
            function_entries.append(entry)

    # Global sort by fim_score descending
    function_entries.sort(key=lambda x: x.get("fim_score", 0), reverse=True)

    print(
        f"  Post-processing: {len(function_entries)} function entries generated, "
        f"{skipped_targets}/{total_targets} targets skipped due to validation failure"
    )
    return function_entries


def print_distribution_stats(function_entries: List[Dict[str, Any]]) -> None:
    """
    Print distribution statistics for key numeric metrics across all
    function-level entries.
    """
    metrics = ["loc", "complexity", "inferability", "fim_score", "difficulty"]

    def _percentile(sorted_vals: List[float], p: float) -> float:
        """Linear interpolation percentile on a pre-sorted list."""
        if not sorted_vals:
            return 0.0
        idx = p / 100.0 * (len(sorted_vals) - 1)
        lower = int(math.floor(idx))
        upper = min(lower + 1, len(sorted_vals) - 1)
        frac = idx - lower
        return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac

    print()
    print("=" * 85)
    print("  Distribution Statistics  (function-level entries)")
    print("=" * 85)

    for metric in metrics:
        values = sorted(
            [float(e[metric]) for e in function_entries if metric in e]
        )
        if not values:
            print(f"\n  {metric}: (no data)")
            continue

        n = len(values)
        mean_val = sum(values) / n
        if n % 2 == 1:
            median_val = values[n // 2]
        else:
            median_val = (values[n // 2 - 1] + values[n // 2]) / 2.0
        variance = sum((v - mean_val) ** 2 for v in values) / n
        std_val = math.sqrt(variance)

        p10 = _percentile(values, 10)
        p25 = _percentile(values, 25)
        p50 = _percentile(values, 50)
        p75 = _percentile(values, 75)
        p90 = _percentile(values, 90)

        print(f"\n  {metric}:")
        print(f"    count  = {n}")
        print(f"    min    = {values[0]:.4f}    max    = {values[-1]:.4f}")
        print(f"    mean   = {mean_val:.4f}    std    = {std_val:.4f}")
        print(
            f"    p10    = {p10:.4f}    p25    = {p25:.4f}    "
            f"p50    = {p50:.4f}    p75    = {p75:.4f}    p90    = {p90:.4f}"
        )

    print()
    print("=" * 85)


# ================================================================
# Pipeline
# ================================================================

def process_samples(
    samples: List[Dict[str, Any]],
    config: Optional[Dict] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Main entry point.  For each sample dict:
      1. file-level pre-filter (line count bounds)
      2. parse code_content & build dependency graph
      3. score & select FIM targets
      4. attach 'mask_targets' list to the dict

    Returns a new list (input is not mutated).
    """
    merged_cfg = {**FIMSelector.DEFAULT_CONFIG, **(config or {})}
    min_file_lines = merged_cfg["min_file_lines"]
    max_file_lines = merged_cfg["max_file_lines"]

    results: List[Dict[str, Any]] = []
    skipped_too_long = 0
    skipped_too_short = 0
    total_selected_functions = 0

    for idx, sample in tqdm(enumerate(samples)):
        out = dict(sample)  # shallow copy
        code = out.get("code_content", "")
        sid = out.get("sample_id", idx)

        # ---- empty / missing code ----
        if not code or not code.strip():
            out["mask_targets"] = []
            out["mask_target_count"] = 0
            out["skip_reason"] = "empty_code"
            results.append(out)
            if verbose:
                print(f"  [{sid}] Empty code_content → skipped")
            continue

        # ---- file-level line count filter ----
        file_line_count = len(code.splitlines())

        if file_line_count < min_file_lines:
            out["mask_targets"] = []
            out["mask_target_count"] = 0
            out["file_lines"] = file_line_count
            out["skip_reason"] = f"too_short ({file_line_count} < {min_file_lines})"
            results.append(out)
            skipped_too_short += 1
            if verbose:
                print(
                    f"  [{sid}] File too short "
                    f"({file_line_count} lines < {min_file_lines}) → skipped"
                )
            continue

        if file_line_count > max_file_lines:
            out["mask_targets"] = []
            out["mask_target_count"] = 0
            out["file_lines"] = file_line_count
            out["skip_reason"] = f"too_long ({file_line_count} > {max_file_lines})"
            results.append(out)
            skipped_too_long += 1
            if verbose:
                print(
                    f"  [{sid}] File too long "
                    f"({file_line_count} lines > {max_file_lines}) → skipped"
                )
            continue

        # ---- build dependency graph ----
        builder = DependencyGraphBuilder(code)
        if not builder.build():
            out["mask_targets"] = []
            out["mask_target_count"] = 0
            out["file_lines"] = file_line_count
            out["skip_reason"] = "syntax_error"
            results.append(out)
            if verbose:
                print(f"  [{sid}] SyntaxError → skipped")
            continue

        if not builder.functions:
            out["mask_targets"] = []
            out["mask_target_count"] = 0
            out["file_lines"] = file_line_count
            out["skip_reason"] = "no_functions"
            results.append(out)
            if verbose:
                print(f"  [{sid}] No functions found → skipped")
            continue

        # ---- select targets ----
        selector = FIMSelector(
            builder.functions, builder.edges, builder.ast_nodes, config
        )
        candidates = selector.select_targets()
        total_selected_functions += len(candidates)

        # ---- serialize into output dict ----
        targets = [
            {
                "func_name": c.func_name,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "body_lineno": c.body_lineno,
                "func_content": c.source_text,
                "loc": c.loc,
                "complexity": c.complexity,
                "inferability": c.inferability,
                "fim_score": c.fim_score,
                "difficulty": c.difficulty,
            }
            for c in candidates
        ]

        out["mask_targets"] = targets
        out["mask_target_count"] = len(targets)
        out["file_lines"] = file_line_count
        out["graph_stats"] = {
            "total_functions": len(builder.functions),
            "call_edges": sum(1 for _, _, t in builder.edges if t == "call"),
            "sibling_pairs": sum(
                1 for _, _, t in builder.edges if t == "sibling"
            ) // 2,
        }
        results.append(out)

        if verbose:
            n_funcs = len(builder.functions)
            n_sel = len(targets)
            print(
                f"  [{sid}] {file_line_count} lines, "
                f"{n_funcs} functions found, "
                f"{n_sel} selected as mask targets"
            )
            for t in targets:
                print(
                    f"      {t['func_name']:35s}  LOC={t['loc']:<3d}  "
                    f"score={t['fim_score']:.4f}  "
                    f"Ĥ={t['complexity']:.3f}  "
                    f"Î={t['inferability']:.3f}  "
                    f"diff={t['difficulty']:.3f}"
                )

    if verbose:
        print()
        print(
            f"  File-level filter summary: "
            f"{skipped_too_short} too short, "
            f"{skipped_too_long} too long, "
            f"{len(results) - skipped_too_short - skipped_too_long} processed"
        )
    print("total_selected_functions", total_selected_functions)
    return results


def save_results(results: List[Dict[str, Any]], path: str):
    """Write results list to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} samples → {path}")