#!/usr/bin/env python3
"""
depfim_multi.py — Multi-Function FIM Target Selection

Extension of depfim.py that selects **groups** of 2 or 3 related functions
as joint FIM mask targets. Designed for mid-training data that improves
code agent capabilities (targeting SWE-bench, SWT-bench, Commit-0).

═══════════════════════════════════════════════════════════════════════
Motivation
═══════════════════════════════════════════════════════════════════════
Real-world code patches (e.g., in SWE-bench) typically modify multiple
related functions simultaneously: fixing a function and updating its
caller, adding a method and wiring it into another, etc. Single-function
FIM trains the model to complete one function given the rest of the file.
Multi-function FIM trains the model to maintain **cross-function
consistency** — interface contracts, shared state, call-chain coherence
— which is crucial for code agent tasks.

═══════════════════════════════════════════════════════════════════════
Group Topology Types
═══════════════════════════════════════════════════════════════════════

Pair (2 functions):
  1. Caller-Callee    : A calls B. Mask both → model must infer both
                        ends of the interface contract.
  2. Co-Callee        : C calls both A and B (C kept unmasked). Mask A
                        and B → model infers two functions from how a
                        shared caller uses them.
  3. Sibling-Coupled  : Same-class methods sharing instance variables
                        (self.xxx). Mask both → model maintains state
                        read/write consistency.
  4. Mutual-Call      : A calls B AND B calls A (direct mutual
                        recursion / delegation). Strongest coupling.

Triple (3 functions):
  1. Call-Chain        : A→B→C. Mask all → model reasons across
                        multiple abstraction layers.
  2. Hub              : A calls B and A calls C. Mask all → model
                        implements an orchestrator and its workers.
  3. Fan-In           : B calls A and C calls A. Mask all → model
                        implements a shared utility and two consumers.
  4. Class-Triad      : Three same-class methods all sharing instance
                        variables. Mask all → model implements a
                        coherent class core.

═══════════════════════════════════════════════════════════════════════
Group-Level Scoring
═══════════════════════════════════════════════════════════════════════

For a group G = {f1, f2, ...}:

  Coupling(G)   ∈ [0, 1]  — normalized count of intra-group edges
                              (call + sibling + shared-variable bonus)
  GroupĤ(G)     — mean of individual Ĥ(fi) scores (not sum, to avoid
                   bias toward larger groups)
  GroupÎ(G)     — re-computed group inferability: the information the
                   *remaining* (unmasked) code provides about the group
                   as a whole. Intra-group signals are subtracted because
                   they disappear when the group is masked together.
  LOC_ratio     — total masked LOC / file LOC (capped at 30% for pairs,
                   40% for triples)

  GroupScore = Coupling × GroupĤ × (GroupÎ / (GroupĤ + GroupÎ + ε))
               × difficulty_penalty

The difficulty penalty applies the same one-sided Gaussian as the
single-function version, operating on group-level difficulty.

═══════════════════════════════════════════════════════════════════════
Usage
═══════════════════════════════════════════════════════════════════════
    python multi_function/step_3_select_function_groups.py

Reads step 2's extracted_python_files.json and writes:
  <out>.json         one record per source file, with a `multi_mask_targets` list
  <out>_groups.json  one record per selected group, each carrying the file with
                     ALL of the group's bodies masked — this is what step 4 eats.

Builds on common/dep_graph.py for the dependency graph and the per-function
Ĥ / Î scores.
"""

import argparse
import ast
import math
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The dependency graph and single-function scoring are shared with the
# single-function pipeline — see common/dep_graph.py.
from common.config import (  # noqa: E402
    add_config_arg, derive_paths, load_config, selection_config,
)
from common.dep_graph import (  # noqa: E402
    DependencyGraphBuilder,
    FIMSelector,
    FunctionInfo,
    mask_function_body,
    save_results,
)


# ================================================================
# Data Structures
# ================================================================

@dataclass
class FunctionGroupCandidate:
    """A group of 2-3 functions selected as joint FIM mask targets."""
    group_type: str              # e.g. "caller_callee", "call_chain"
    group_size: int              # 2 or 3
    functions: List[Dict]        # per-function info dicts
    total_loc: int
    loc_ratio: float             # total masked LOC / file LOC
    coupling: float              # intra-group coupling score
    group_complexity: float      # mean Ĥ of group members
    group_inferability: float    # re-computed group-level Î
    group_score: float           # final composite score
    group_difficulty: float      # group-level difficulty


# ================================================================
# Shared Instance Variable Analyzer
# ================================================================

class SharedStateAnalyzer:
    """
    Simple shared-instance-variable analysis.
    For each method, collect the set of `self.xxx` attribute names
    accessed (both read and write). Two methods "share state" if
    their self-attribute sets have a non-empty intersection.
    """

    def __init__(self, ast_nodes: Dict[str, ast.AST],
                 functions: Dict[str, FunctionInfo]):
        self.ast_nodes = ast_nodes
        self.functions = functions
        # fname -> set of self.attr names
        self._self_attrs: Dict[str, Set[str]] = {}
        self._analyze_all()

    def _analyze_all(self):
        for fname, node in self.ast_nodes.items():
            fi = self.functions.get(fname)
            if fi is None or fi.class_name is None:
                continue
            attrs: Set[str] = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Attribute):
                    if (isinstance(child.value, ast.Name)
                            and child.value.id in ("self", "cls")):
                        attrs.add(child.attr)
            self._self_attrs[fname] = attrs

    def get_attrs(self, fname: str) -> Set[str]:
        return self._self_attrs.get(fname, set())

    def shared_attrs(self, a: str, b: str) -> Set[str]:
        return self.get_attrs(a) & self.get_attrs(b)

    def sharing_ratio(self, a: str, b: str) -> float:
        """Jaccard-like ratio of shared self-attributes."""
        sa, sb = self.get_attrs(a), self.get_attrs(b)
        union = sa | sb
        if not union:
            return 0.0
        return len(sa & sb) / len(union)


# ================================================================
# Group Enumeration — Pair Patterns
# ================================================================

class GroupEnumerator:
    """
    Enumerate candidate function groups based on dependency topology.
    Requires at least one direct edge between group members (Decision Q1=A).
    """

    def __init__(
        self,
        functions: Dict[str, FunctionInfo],
        edges: List[Tuple[str, str, str]],
        ast_nodes: Dict[str, ast.AST],
        state_analyzer: SharedStateAnalyzer,
    ):
        self.functions = functions
        self.ast_nodes = ast_nodes
        self.state = state_analyzer

        # Build adjacency maps
        self.call_fwd: Dict[str, Set[str]] = defaultdict(set)   # caller -> callees
        self.call_rev: Dict[str, Set[str]] = defaultdict(set)   # callee -> callers
        self.siblings: Dict[str, Set[str]] = defaultdict(set)
        for src, dst, etype in edges:
            if etype == "call":
                self.call_fwd[src].add(dst)
                self.call_rev[dst].add(src)
            elif etype == "sibling":
                self.siblings[src].add(dst)

        # All function names as a set for quick membership checks
        self._all_fnames = set(functions.keys())

    # ----------------------------------------------------------------
    # Pair enumeration
    # ----------------------------------------------------------------

    def enumerate_pairs(self) -> List[Tuple[str, str, str]]:
        """
        Return list of (funcA, funcB, pattern_type) tuples.
        Each pair has at least one direct edge.
        """
        pairs: List[Tuple[str, str, str]] = []
        seen: Set[frozenset] = set()

        # Pattern 1: Caller-Callee
        for caller in self._all_fnames:
            for callee in self.call_fwd.get(caller, set()):
                if callee not in self._all_fnames:
                    continue
                key = frozenset((caller, callee))
                if key not in seen:
                    seen.add(key)
                    pairs.append((caller, callee, "caller_callee"))

        # Pattern 2: Co-Callee (A and B both called by some C, C not in group)
        # For each function C, look at pairs among its callees
        for c in self._all_fnames:
            callees_of_c = [
                x for x in self.call_fwd.get(c, set())
                if x in self._all_fnames and x != c
            ]
            for i, a in enumerate(callees_of_c):
                for b in callees_of_c[i + 1:]:
                    key = frozenset((a, b))
                    if key not in seen:
                        seen.add(key)
                        pairs.append((a, b, "co_callee"))

        # Pattern 3: Sibling-Coupled (same class + shared self.attrs)
        for a in self._all_fnames:
            fi_a = self.functions[a]
            if fi_a.class_name is None:
                continue
            for b in self.siblings.get(a, set()):
                if b not in self._all_fnames or b <= a:
                    continue  # b <= a avoids duplicates for undirected sibling
                shared = self.state.shared_attrs(a, b)
                if shared:  # must share at least one instance variable
                    key = frozenset((a, b))
                    if key not in seen:
                        seen.add(key)
                        pairs.append((a, b, "sibling_coupled"))

        # Pattern 4: Mutual-Call (A calls B AND B calls A)
        for a in self._all_fnames:
            for b in self.call_fwd.get(a, set()):
                if b not in self._all_fnames:
                    continue
                if a in self.call_fwd.get(b, set()):
                    key = frozenset((a, b))
                    if key not in seen:
                        seen.add(key)
                        pairs.append((a, b, "mutual_call"))

        return pairs

    # ----------------------------------------------------------------
    # Triple enumeration
    # ----------------------------------------------------------------

    def enumerate_triples(self) -> List[Tuple[str, str, str, str]]:
        """
        Return list of (funcA, funcB, funcC, pattern_type) tuples.
        Each triple has direct edges forming the expected topology.
        """
        triples: List[Tuple[str, str, str, str]] = []
        seen: Set[frozenset] = set()

        # Pattern 1: Call-Chain  A→B→C
        for a in self._all_fnames:
            for b in self.call_fwd.get(a, set()):
                if b not in self._all_fnames or b == a:
                    continue
                for c in self.call_fwd.get(b, set()):
                    if c not in self._all_fnames or c in (a, b):
                        continue
                    key = frozenset((a, b, c))
                    if key not in seen:
                        seen.add(key)
                        triples.append((a, b, c, "call_chain"))

        # Pattern 2: Hub  A→B, A→C
        for a in self._all_fnames:
            callees_of_a = [
                x for x in self.call_fwd.get(a, set())
                if x in self._all_fnames and x != a
            ]
            for i, b in enumerate(callees_of_a):
                for c in callees_of_a[i + 1:]:
                    key = frozenset((a, b, c))
                    if key not in seen:
                        seen.add(key)
                        triples.append((a, b, c, "hub"))

        # Pattern 3: Fan-In  B→A, C→A
        for a in self._all_fnames:
            callers_of_a = [
                x for x in self.call_rev.get(a, set())
                if x in self._all_fnames and x != a
            ]
            for i, b in enumerate(callers_of_a):
                for c in callers_of_a[i + 1:]:
                    key = frozenset((a, b, c))
                    if key not in seen:
                        seen.add(key)
                        triples.append((a, b, c, "fan_in"))

        # Pattern 4: Class-Triad  (same class, all share instance vars)
        cls_methods: Dict[str, List[str]] = defaultdict(list)
        for fn, fi in self.functions.items():
            if fi.class_name:
                cls_methods[fi.class_name].append(fn)

        for cls_name, methods in cls_methods.items():
            if len(methods) < 3:
                continue
            for combo in combinations(methods, 3):
                a, b, c = combo
                # All three must pairwise share at least one self.attr
                if (self.state.shared_attrs(a, b)
                        and self.state.shared_attrs(b, c)
                        and self.state.shared_attrs(a, c)):
                    key = frozenset(combo)
                    if key not in seen:
                        seen.add(key)
                        triples.append((a, b, c, "class_triad"))

        return triples


# ================================================================
# Group Scorer
# ================================================================

class GroupScorer:
    """
    Score function groups using coupling, complexity, and
    re-computed group-level inferability.
    """

    DEFAULT_GROUP_CONFIG = {
        # --- LOC ratio caps ---
        "pair_max_loc_ratio": 0.30,
        "triple_max_loc_ratio": 0.40,
        # --- Per-function LOC bounds (same as single) ---
        "min_loc": 10,
        "max_loc": 200,
        # --- Coupling weights ---
        "coupling_w_call": 0.50,        # weight for call edges within group
        "coupling_w_sibling": 0.20,     # weight for sibling edges
        "coupling_w_shared_state": 0.30, # weight for shared instance vars
        # --- Group score thresholds ---
        "pair_score_threshold": 0.04,
        "triple_score_threshold": 0.03,
        "min_coupling": 0.15,
        "min_group_complexity": 0.15,
        # --- Difficulty penalty (same scheme as single) ---
        "difficulty_ceiling": 0.55,
        "difficulty_sigma": 0.20,
        # --- Max groups per file ---
        "max_pairs_per_file": 5,
        "max_triples_per_file": 3,
    }

    # Same dunder skip list as single-function version
    SKIP_NAMES = FIMSelector.SKIP_NAMES

    def __init__(
        self,
        functions: Dict[str, FunctionInfo],
        edges: List[Tuple[str, str, str]],
        ast_nodes: Dict[str, ast.AST],
        state_analyzer: SharedStateAnalyzer,
        single_selector: FIMSelector,
        file_line_count: int,
        config: Optional[Dict] = None,
    ):
        self.functions = functions
        self.ast_nodes = ast_nodes
        self.state = state_analyzer
        self.single = single_selector
        self.file_lines = file_line_count
        self.cfg = {**self.DEFAULT_GROUP_CONFIG, **(config or {})}

        # Adjacency (reuse from single selector)
        self.callers = single_selector.callers
        self.callees = single_selector.callees
        self.siblings = single_selector.siblings

        # Call edges as a set for O(1) lookup
        self._call_edge_set: Set[Tuple[str, str]] = set()
        self._sibling_edge_set: Set[frozenset] = set()
        for src, dst, etype in edges:
            if etype == "call":
                self._call_edge_set.add((src, dst))
            elif etype == "sibling":
                self._sibling_edge_set.add(frozenset((src, dst)))

    # ----------------------------------------------------------------
    # Pre-filter: check if individual functions pass basic criteria
    # ----------------------------------------------------------------

    def _passes_individual_filter(self, fname: str) -> bool:
        fi = self.functions[fname]
        if fi.loc < self.cfg["min_loc"] or fi.loc > self.cfg["max_loc"]:
            return False
        if fname.split(".")[-1] in self.SKIP_NAMES:
            return False
        return True

    # ----------------------------------------------------------------
    # Coupling score
    # ----------------------------------------------------------------

    def compute_coupling(self, group: List[str]) -> float:
        """
        Normalized coupling among group members.

        Components:
          - call edges within the group
          - sibling edges within the group
          - shared instance variable ratio (pairwise mean)
        """
        c = self.cfg
        n = len(group)
        max_directed = n * (n - 1)       # max possible directed call edges
        max_undirected = n * (n - 1) / 2  # max possible undirected pairs

        # Count intra-group call edges
        call_count = 0
        for i, a in enumerate(group):
            for b in group:
                if a != b and (a, b) in self._call_edge_set:
                    call_count += 1
        call_score = call_count / max(max_directed, 1)

        # Count intra-group sibling edges
        sib_count = 0
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if frozenset((a, b)) in self._sibling_edge_set:
                    sib_count += 1
        sib_score = sib_count / max(max_undirected, 1)

        # Pairwise shared-state ratio
        state_ratios = []
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                state_ratios.append(self.state.sharing_ratio(a, b))
        state_score = (sum(state_ratios) / len(state_ratios)) if state_ratios else 0.0

        return (
            c["coupling_w_call"] * min(call_score, 1.0)
            + c["coupling_w_sibling"] * min(sib_score, 1.0)
            + c["coupling_w_shared_state"] * min(state_score, 1.0)
        )

    # ----------------------------------------------------------------
    # Group complexity (mean of individual Ĥ)
    # ----------------------------------------------------------------

    def compute_group_complexity(self, group: List[str]) -> float:
        complexities = [
            self.single.compute_complexity(self.functions[fn])
            for fn in group
        ]
        return sum(complexities) / len(complexities)

    # ----------------------------------------------------------------
    # Group inferability (re-computed with intra-group info subtracted)
    # ----------------------------------------------------------------

    def compute_group_inferability(self, group: List[str]) -> float:
        """
        Re-compute inferability as if ALL group members are masked.
        Intra-group call/sibling information disappears.
        """
        group_set = set(group)
        total_inf = 0.0

        for fname in group:
            fi = self.functions[fname]
            c = self.single.cfg

            # 1) Caller information — only from callers OUTSIDE the group
            caller_sp = 0.0
            for caller in self.callers.get(fname, set()):
                if caller in self.functions and caller not in group_set:
                    caller_sp += self.single._call_site_specificity(caller, fname)
            caller_score = min(caller_sp / 3.0, 1.0)

            # 2) Callee information — only callees OUTSIDE the group
            internal_callees = [
                x for x in self.callees.get(fname, set())
                if x in self.functions and x not in group_set
            ]
            callee_score = min(len(internal_callees) / 4.0, 1.0)

            # 3) Signature information (unchanged — signatures are kept)
            sig_score = 0.0
            if fi.has_return_type:
                sig_score += 0.30
            if fi.has_param_types:
                sig_score += 0.25
            raw_name = fname.split(".")[-1]
            if not raw_name.startswith("_"):
                parts = [p for p in raw_name.split("_") if p]
                sig_score += min(len(parts) / 5.0, 0.25)
            non_self = [p for p in fi.params if p not in ("self", "cls")]
            sig_score += min(len(non_self) / 6.0, 0.20)
            sig_score = min(sig_score, 1.0)

            # 4) Documentation (unchanged — docstrings NOT in body for masking)
            #    NOTE: actually docstring IS part of body, so it IS masked.
            #    For group-level, docstring info vanishes.
            doc_score = 0.0  # masked away with the body

            # 5) Class context — only siblings OUTSIDE the group
            class_score = 0.0
            if fi.class_name:
                ext_sibs = [
                    s for s in self.siblings.get(fname, set())
                    if s not in group_set
                ]
                class_score = min(len(ext_sibs) / 5.0, 1.0)
                init_name = f"{fi.class_name}.__init__"
                if (init_name in self.functions
                        and init_name != fname
                        and init_name not in group_set):
                    class_score = min(class_score + 0.3, 1.0)

            ind_inf = (
                c["alpha_caller"] * caller_score
                + c["beta_callee"] * callee_score
                + c["gamma_sig"] * sig_score
                + c["delta_doc"] * doc_score
                + c["epsilon_class"] * class_score
            )
            total_inf += ind_inf

        return total_inf / len(group)

    # ----------------------------------------------------------------
    # Full group scoring
    # ----------------------------------------------------------------

    def score_group(
        self, group: List[str], pattern: str
    ) -> Optional[FunctionGroupCandidate]:
        """
        Compute all metrics and return a FunctionGroupCandidate,
        or None if the group fails any filter.
        """
        c = self.cfg
        n = len(group)

        # --- Individual filters ---
        for fn in group:
            if not self._passes_individual_filter(fn):
                return None

        # --- LOC ratio ---
        total_loc = sum(self.functions[fn].loc for fn in group)
        loc_ratio = total_loc / max(self.file_lines, 1)
        max_ratio = (c["pair_max_loc_ratio"] if n == 2
                     else c["triple_max_loc_ratio"])
        if loc_ratio > max_ratio:
            return None

        # --- Coupling ---
        coupling = self.compute_coupling(group)
        if coupling < c["min_coupling"]:
            return None

        # --- Complexity ---
        group_h = self.compute_group_complexity(group)
        if group_h < c["min_group_complexity"]:
            return None

        # --- Inferability ---
        group_i = self.compute_group_inferability(group)

        # --- Group score ---
        raw_score = coupling * group_h * (group_i / (group_h + group_i + 1e-8))

        # --- Difficulty + penalty ---
        difficulty = max(0.0, group_h - group_i) / (group_h + 1e-8)
        if difficulty > c["difficulty_ceiling"]:
            excess = difficulty - c["difficulty_ceiling"]
            factor = math.exp(
                -(excess ** 2) / (2.0 * c["difficulty_sigma"] ** 2)
            )
        else:
            factor = 1.0
        final_score = raw_score * factor

        # --- Threshold ---
        threshold = (c["pair_score_threshold"] if n == 2
                     else c["triple_score_threshold"])
        if final_score < threshold:
            return None

        # --- Build per-function info ---
        func_dicts = []
        for fn in group:
            fi = self.functions[fn]
            func_dicts.append({
                "func_name": fn,
                "start_line": fi.lineno,
                "end_line": fi.end_lineno,
                "body_lineno": fi.body_lineno,
                "func_content": fi.source_text,
                "loc": fi.loc,
                "complexity": round(self.single.compute_complexity(fi), 4),
                "inferability": round(self.single.compute_inferability(fn), 4),
            })

        return FunctionGroupCandidate(
            group_type=pattern,
            group_size=n,
            functions=func_dicts,
            total_loc=total_loc,
            loc_ratio=round(loc_ratio, 4),
            coupling=round(coupling, 4),
            group_complexity=round(group_h, 4),
            group_inferability=round(group_i, 4),
            group_score=round(final_score, 4),
            group_difficulty=round(difficulty, 4),
        )

    # ----------------------------------------------------------------
    # Select top groups per file
    # ----------------------------------------------------------------

    def select_groups(
        self,
        pairs: List[Tuple[str, str, str]],
        triples: List[Tuple[str, str, str, str]],
    ) -> List[FunctionGroupCandidate]:
        """
        Score all candidate groups, apply overlap dedup, return top-K.
        """
        c = self.cfg
        all_candidates: List[FunctionGroupCandidate] = []

        # Score pairs
        for a, b, pattern in pairs:
            cand = self.score_group([a, b], pattern)
            if cand is not None:
                all_candidates.append(cand)

        # Score triples
        for *funcs, pattern in triples:
            cand = self.score_group(list(funcs), pattern)
            if cand is not None:
                all_candidates.append(cand)

        # Sort by score descending
        all_candidates.sort(key=lambda x: x.group_score, reverse=True)

        # Greedy dedup: once a function is in a selected group, it cannot
        # appear in another group (avoids overlapping masks)
        selected: List[FunctionGroupCandidate] = []
        used_fns: Set[str] = set()
        pair_count = 0
        triple_count = 0

        for cand in all_candidates:
            fns = {fd["func_name"] for fd in cand.functions}
            if fns & used_fns:
                continue  # overlap with already selected group

            if cand.group_size == 2:
                if pair_count >= c["max_pairs_per_file"]:
                    continue
                pair_count += 1
            else:
                if triple_count >= c["max_triples_per_file"]:
                    continue
                triple_count += 1

            selected.append(cand)
            used_fns.update(fns)

        return selected


# ================================================================
# Multi-mask utility
# ================================================================

def mask_function_group(
    code_content: str,
    functions: List[Dict],
) -> str:
    """
    Mask multiple function bodies independently.
    Functions are processed from bottom to top (by start_line desc)
    to avoid line-number shifts.

    Each function body is replaced with:
        # <MASKED_FUNCTION_BODY>  (at the correct indentation)
    """
    # Sort by start_line descending so later functions are masked first
    sorted_fns = sorted(functions, key=lambda f: f["start_line"], reverse=True)

    result = code_content
    for fn in sorted_fns:
        result = mask_function_body(
            result,
            fn["start_line"],
            fn["end_line"],
            fn["body_lineno"],
        )
    return result


# ================================================================
# Post-processing (multi-function version)
# ================================================================

def postprocess_multi_results(
    results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Expand sample-level results into group-level entries.

    Each group becomes its own dict containing:
      - sample-level fields (minus internals)
      - group-level fields (type, score, coupling, etc.)
      - per-function details in 'functions' list
      - masked_code_content with all group functions masked
    """
    group_entries: List[Dict[str, Any]] = []
    total_groups = 0
    skipped_groups = 0

    for sample in results:
        groups = sample.get("multi_mask_targets", [])
        if not groups:
            continue

        code_content = sample.get("code_content", "")
        code_lines = code_content.splitlines()

        sample_fields = {
            k: v for k, v in sample.items()
            if k not in (
                "multi_mask_targets", "multi_mask_target_count",
                "skip_reason", "code_content",
                # also exclude single-function fields if present
                "mask_targets", "mask_target_count",
            )
        }

        for group in groups:
            total_groups += 1

            # Validate each function's source text
            valid = True
            for fn in group.get("functions", []):
                start = fn["start_line"]
                end = fn["end_line"]
                expected = fn.get("func_content", "")
                extracted = "\n".join(code_lines[start - 1: end])

                if extracted != expected:
                    # Lenient whitespace check
                    e_s = "\n".join(l.rstrip() for l in code_lines[start - 1: end])
                    x_s = "\n".join(l.rstrip() for l in expected.splitlines())
                    if e_s != x_s:
                        sid = sample.get("sample_id", "?")
                        fname = fn.get("func_name", "?")
                        print(
                            f"  [WARN] sample {sid}, group func {fname}: "
                            f"source mismatch → group skipped"
                        )
                        valid = False
                        break

            if not valid:
                skipped_groups += 1
                continue

            entry = dict(sample_fields)
            entry["group_type"] = group["group_type"]
            entry["group_size"] = group["group_size"]
            entry["functions"] = group["functions"]
            entry["total_loc"] = group["total_loc"]
            entry["loc_ratio"] = group["loc_ratio"]
            entry["coupling"] = group["coupling"]
            entry["group_complexity"] = group["group_complexity"]
            entry["group_inferability"] = group["group_inferability"]
            entry["group_score"] = group["group_score"]
            entry["group_difficulty"] = group["group_difficulty"]

            entry["masked_code_content"] = mask_function_group(
                code_content, group["functions"]
            )
            group_entries.append(entry)

    group_entries.sort(key=lambda x: x.get("group_score", 0), reverse=True)

    print(
        f"  Multi post-processing: {len(group_entries)} group entries generated, "
        f"{skipped_groups}/{total_groups} groups skipped"
    )
    return group_entries


def print_group_distribution_stats(
    group_entries: List[Dict[str, Any]]
) -> None:
    """Print distribution stats for multi-function group entries."""
    if not group_entries:
        print("\n  No multi-function group entries to report.\n")
        return

    # --- Type breakdown ---
    type_counts: Dict[str, int] = defaultdict(int)
    size_counts: Dict[int, int] = defaultdict(int)
    for e in group_entries:
        type_counts[e.get("group_type", "?")] += 1
        size_counts[e.get("group_size", 0)] += 1

    print()
    print("=" * 85)
    print("  Multi-Function Group Distribution Statistics")
    print("=" * 85)

    print("\n  Group type breakdown:")
    for t, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:25s}  {cnt}")

    print(f"\n  Group size breakdown:")
    for sz, cnt in sorted(size_counts.items()):
        print(f"    size={sz}  →  {cnt} groups")

    # --- Numeric metrics ---
    metrics = [
        "total_loc", "loc_ratio", "coupling",
        "group_complexity", "group_inferability",
        "group_score", "group_difficulty",
    ]

    def _percentile(sorted_vals: List[float], p: float) -> float:
        if not sorted_vals:
            return 0.0
        idx = p / 100.0 * (len(sorted_vals) - 1)
        lower = int(math.floor(idx))
        upper = min(lower + 1, len(sorted_vals) - 1)
        frac = idx - lower
        return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac

    for metric in metrics:
        values = sorted(
            [float(e[metric]) for e in group_entries if metric in e]
        )
        if not values:
            print(f"\n  {metric}: (no data)")
            continue

        n = len(values)
        mean_val = sum(values) / n
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
# Main Pipeline
# ================================================================

def process_samples_multi(
    samples: List[Dict[str, Any]],
    config: Optional[Dict] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Process each sample and select multi-function FIM groups.

    For each sample:
      1. File-level pre-filter
      2. Build dependency graph
      3. Analyze shared instance variables
      4. Enumerate candidate pairs and triples
      5. Score and select top groups
      6. Attach 'multi_mask_targets' to the output dict
    """
    merged_cfg = {
        **FIMSelector.DEFAULT_CONFIG,
        **GroupScorer.DEFAULT_GROUP_CONFIG,
        **(config or {}),
    }
    min_file_lines = merged_cfg["min_file_lines"]
    max_file_lines = merged_cfg["max_file_lines"]

    results: List[Dict[str, Any]] = []
    skipped_short = 0
    skipped_long = 0
    total_pairs = 0
    total_triples = 0

    for idx, sample in tqdm(enumerate(samples)):
        out = dict(sample)
        code = out.get("code_content", "")
        sid = out.get("sample_id", idx)

        # ---- empty code ----
        if not code or not code.strip():
            out["multi_mask_targets"] = []
            out["multi_mask_target_count"] = 0
            out["skip_reason"] = "empty_code"
            results.append(out)
            continue

        # ---- file-level filter ----
        file_line_count = len(code.splitlines())

        if file_line_count < min_file_lines:
            out["multi_mask_targets"] = []
            out["multi_mask_target_count"] = 0
            out["file_lines"] = file_line_count
            out["skip_reason"] = f"too_short ({file_line_count} < {min_file_lines})"
            results.append(out)
            skipped_short += 1
            continue

        if file_line_count > max_file_lines:
            out["multi_mask_targets"] = []
            out["multi_mask_target_count"] = 0
            out["file_lines"] = file_line_count
            out["skip_reason"] = f"too_long ({file_line_count} > {max_file_lines})"
            results.append(out)
            skipped_long += 1
            continue

        # ---- build dependency graph ----
        builder = DependencyGraphBuilder(code)
        if not builder.build():
            out["multi_mask_targets"] = []
            out["multi_mask_target_count"] = 0
            out["file_lines"] = file_line_count
            out["skip_reason"] = "syntax_error"
            results.append(out)
            continue

        if len(builder.functions) < 2:
            out["multi_mask_targets"] = []
            out["multi_mask_target_count"] = 0
            out["file_lines"] = file_line_count
            out["skip_reason"] = "insufficient_functions"
            results.append(out)
            continue

        # ---- shared state analysis ----
        state_analyzer = SharedStateAnalyzer(builder.ast_nodes, builder.functions)

        # ---- single selector (reuse for individual Ĥ/Î computation) ----
        single_selector = FIMSelector(
            builder.functions, builder.edges, builder.ast_nodes, config
        )

        # ---- enumerate candidates ----
        enumerator = GroupEnumerator(
            builder.functions, builder.edges, builder.ast_nodes, state_analyzer
        )
        pairs = enumerator.enumerate_pairs()
        triples = enumerator.enumerate_triples() if len(builder.functions) >= 3 else []

        # ---- score & select ----
        scorer = GroupScorer(
            builder.functions, builder.edges, builder.ast_nodes,
            state_analyzer, single_selector, file_line_count, config,
        )
        selected = scorer.select_groups(pairs, triples)

        n_pairs = sum(1 for g in selected if g.group_size == 2)
        n_triples = sum(1 for g in selected if g.group_size == 3)
        total_pairs += n_pairs
        total_triples += n_triples

        # ---- serialize ----
        targets = []
        for g in selected:
            targets.append({
                "group_type": g.group_type,
                "group_size": g.group_size,
                "functions": g.functions,
                "total_loc": g.total_loc,
                "loc_ratio": g.loc_ratio,
                "coupling": g.coupling,
                "group_complexity": g.group_complexity,
                "group_inferability": g.group_inferability,
                "group_score": g.group_score,
                "group_difficulty": g.group_difficulty,
            })

        out["multi_mask_targets"] = targets
        out["multi_mask_target_count"] = len(targets)
        out["file_lines"] = file_line_count
        out["multi_graph_stats"] = {
            "total_functions": len(builder.functions),
            "candidate_pairs": len(pairs),
            "candidate_triples": len(triples),
            "selected_pairs": n_pairs,
            "selected_triples": n_triples,
        }
        results.append(out)

        if verbose:
            print(
                f"  [{sid}] {file_line_count} lines, "
                f"{len(builder.functions)} funcs, "
                f"pairs: {len(pairs)} candidates → {n_pairs} selected, "
                f"triples: {len(triples)} candidates → {n_triples} selected"
            )
            for g in selected:
                fn_names = [f["func_name"] for f in g.functions]
                print(
                    f"      [{g.group_type}] {fn_names}  "
                    f"LOC={g.total_loc}  ratio={g.loc_ratio:.2f}  "
                    f"coupling={g.coupling:.3f}  "
                    f"score={g.group_score:.4f}  "
                    f"Ĥ={g.group_complexity:.3f}  "
                    f"Î={g.group_inferability:.3f}  "
                    f"diff={g.group_difficulty:.3f}"
                )

    print(
        f"\n  Multi-function summary: "
        f"{skipped_short} too short, {skipped_long} too long, "
        f"{len(results) - skipped_short - skipped_long} processed"
    )
    print(f"  Total selected: {total_pairs} pairs, {total_triples} triples")
    return results


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Select coupled groups of 2-3 functions as joint FIM mask targets."
    )
    add_config_arg(parser)
    parser.add_argument("--input", "-i", default=None,
                        help="Override <work_dir>/extracted_python_files.json (step 2 output)")
    parser.add_argument("--output", "-o", default=None,
                        help="Override <work_dir>/multi_function/step_3_selected_groups.json")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress the per-file breakdown")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = derive_paths(cfg)

    input_path = Path(args.input) if args.input else paths["extracted_files"]
    output_path = Path(args.output) if args.output else paths["multi_step3_out"]

    if not input_path.exists():
        sys.exit(
            f"Error: input not found: {input_path}\n"
            "Run common/step_2_extract_python_files.py first."
        )

    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples from {input_path}")

    selection = selection_config(cfg)
    if selection:
        print(f"Selection config: {selection}")

    # ---- Stage 1: multi-function group selection ----
    results = process_samples_multi(samples, config=selection, verbose=not args.quiet)
    save_results(results, str(output_path))

    # ---- Stage 2: group-level post-processing (one entry per group, masked) ----
    group_entries = postprocess_multi_results(results)
    group_output_path = output_path.parent / f"{output_path.stem}_groups{output_path.suffix}"
    save_results(group_entries, str(group_output_path))

    # ---- Stage 3: distribution stats ----
    print_group_distribution_stats(group_entries)

    print(f"\n✅ Step 3 done. Feed step 4 with:\n   {group_output_path}")


if __name__ == "__main__":
    main()

