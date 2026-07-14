#!/usr/bin/env python3
"""
Step 4 (multi-function) — complete a whole group of functions, then critique it.

The multi-function counterpart of single_function/step_4. Instead of one masked
body, the model gets 2-3 masked bodies from the same file that are structurally
coupled (caller-callee, co-callee, call chain, ...), and must complete them
*together*. That is the point of the whole exercise: real patches touch several
functions at once, and the model has to keep the interface contracts, shared
state and call chains consistent across them.

The critique then scores each function individually AND grades the group's
cross-function coherence (interface / state / logic consistency). One bad
function poisons the whole group — the critic is told to discard it.

Two things differ from the single-function step 4:
  - Pre-sharding: with --pre-shard the input is split into per-shard files on
    disk once, so 200 parallel workers don't each load the full dataset into
    memory. Run it once before fanning out.
  - Pairs and triples are written to separate output files.

    # once, before fanning out
    python multi_function/step_4_multi_fim_and_critique.py --pre-shard --total-shards 200

    # then each shard (see scripts/run_step4_parallel.sh)
    python multi_function/step_4_multi_fim_and_critique.py --shard 1 --total-shards 200
"""

import json
import logging
import re
import sys
import threading
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Dict

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(*a, **kw):
        raise SystemExit("tqdm is required. Install it with: pip install tqdm")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import add_config_arg, derive_paths, load_config  # noqa: E402
from common.llm_client import (  # noqa: E402
    WANDB_AVAILABLE, GeminiClient, TokenCounter, finish_wandb, init_wandb,
)

if WANDB_AVAILABLE:
    import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Unique ID Generation
# =============================================================================

def make_unique_id(record: dict) -> str:
    """
    Generate a deterministic unique ID for a multi-function group record.

    Uses sample_id + group_type + sorted function names to ensure uniqueness
    and checkpoint consistency across restarts.
    """
    sample_id = record.get('sample_id', '')
    group_type = record.get('group_type', '')
    functions = record.get('functions', [])
    func_names = sorted(f.get('func_name', '') for f in functions)
    func_key = "|".join(func_names)
    return f"{sample_id}__{group_type}__{func_key}"


# =============================================================================
# Prompt Templates
# =============================================================================

MULTI_FIM_COMPLETION_PROMPT = '''You are an expert Python programmer. Your task is to complete {num_functions} masked functions based on the surrounding code context.

## Task

Below is a Python file where {num_functions} function bodies have been replaced with `# <MASKED_FUNCTION_BODY>`. These functions are **structurally related** — they {relationship_description}.

Your job is to:

1. **Analyze the context**: Look at the imports, other functions, class definitions, and how these functions interact with the rest of the code and with EACH OTHER.
2. **Reason about cross-function consistency**: These functions are related. Think about:
   - How they call each other or are called together
   - Whether they share instance variables or state
   - Whether they need to maintain consistent interfaces (parameter passing, return types)
   - The overall design pattern they implement together
3. **Write the complete function bodies**: Provide working implementations for ALL {num_functions} functions.

## Output Format

Please structure your response as follows:

### Reasoning
<Your step-by-step analysis of what each function should do, and how they relate to each other>

{function_output_sections}

## Important Notes
- Provide the function BODY only (the code inside the function), not the signature
- Maintain proper indentation (the body should be indented with 4 spaces or appropriate level)
- The implementations should be **mutually consistent** — if function A calls function B, make sure the call matches B's actual implementation
- If a function has a docstring, include it as part of the body
- Pay special attention to shared state (e.g., self.xxx attributes) being used consistently across functions

## Code with Masked Functions

```python
{masked_code}
```

## Functions to Complete

{function_list}

Please analyze the context and provide your reasoning followed by the implementations.
'''

MULTI_CRITIQUE_PROMPT = '''You are an expert code reviewer. Your task is to evaluate a **group** of {num_functions} function completions against their ground truth implementations, AND determine whether each function and the group as a whole are suitable for FIM evaluation.

## Context

A model was asked to complete {num_functions} masked functions simultaneously. These functions are related: they {relationship_description}.

## Code with Masked Functions

```python
{masked_code}
```

## Per-Function Evaluation

{per_function_sections}

## Part 1: Per-Function Feasibility & Quality

For EACH function, evaluate:

**Feasibility**: Is the function completable from context alone?
- **Infeasible** if it relies on obscure external APIs, domain-specific conventions not in the file, or magic constants that cannot be guessed.
- **Feasible** if its behavior can be reasonably inferred from the function name, signature, docstring, surrounding code, and the OTHER functions in the group.

**Quality Scores** (1-5 scale):
1. **Correctness**: Does it produce the same results as ground truth?
2. **Executability**: Can it run without errors?
3. **API Usage**: Does it correctly use other functions/methods in the codebase?
4. **Completeness**: Does it handle all cases the ground truth handles?

## Part 2: Group Coherence

Evaluate the **cross-function consistency** of the completions:

1. **Interface Consistency (1-5)**: Do the functions correctly call each other? Do parameter types, return types, and calling conventions match between the completed functions?
2. **State Consistency (1-5)**: If functions share instance variables (self.xxx), are reads and writes consistent? Does one function write what the other reads?
3. **Logic Consistency (1-5)**: Do the functions implement a coherent overall design? Would they work correctly together as a unit?

## Output Format

```json
{{
  "per_function": [
    {{
      "func_name": "<function name>",
      "feasibility": {{
        "is_feasible": <true or false>,
        "confidence": <0.0 to 1.0>,
        "reason": "<explanation>"
      }},
      "scores": {{
        "correctness": <1-5>,
        "correctness_reason": "<brief explanation>",
        "executability": <1-5>,
        "executability_reason": "<brief explanation>",
        "api_usage": <1-5>,
        "api_usage_reason": "<brief explanation>",
        "completeness": <1-5>,
        "completeness_reason": "<brief explanation>"
      }},
      "overall_score": <1-5>,
      "overall_reason": "<summary>"
    }}
  ],
  "group_coherence": {{
    "interface_consistency": <1-5>,
    "interface_consistency_reason": "<explanation>",
    "state_consistency": <1-5>,
    "state_consistency_reason": "<explanation>",
    "logic_consistency": <1-5>,
    "logic_consistency_reason": "<explanation>",
    "coherence_score": <1-5, average of the three above>,
    "coherence_reason": "<overall coherence summary>"
  }},
  "group_overall_score": <1-5, considering both individual quality and coherence>,
  "group_overall_reason": "<summary>",
  "should_discard": <true or false>,
  "discard_reason": "<if should_discard is true, explain why>"
}}
```

## Decision Logic for `should_discard`

Set `should_discard` to `true` if ANY of the following:
1. **Any function is infeasible** — the group contains a function that fundamentally cannot be completed from context.
2. **Any function has executability ≤ 1** — a non-functional completion poisons the group.
3. **Any function has overall_score ≤ 1** — one very bad function makes the group unusable.
4. **Group coherence_score ≤ 2** — the functions don't work together coherently.
5. **group_overall_score ≤ 2** — the group as a whole is too poor.

Otherwise set `should_discard` to `false`.

Please provide your detailed analysis.
'''


# =============================================================================
# Relationship Descriptions for Prompts
# =============================================================================

GROUP_TYPE_DESCRIPTIONS = {
    # Pairs
    "caller_callee": "have a caller-callee relationship (one calls the other)",
    "co_callee": "are both called by the same function (co-callees)",
    "sibling_coupled": "are methods of the same class that share instance variables",
    "mutual_call": "call each other (mutual dependency)",
    # Triples
    "call_chain": "form a call chain (A calls B, B calls C)",
    "hub": "have a hub pattern (one function calls the other two)",
    "fan_in": "have a fan-in pattern (two functions both call the third)",
    "class_triad": "are three methods of the same class sharing instance variables",
}


def get_relationship_description(group_type: str) -> str:
    return GROUP_TYPE_DESCRIPTIONS.get(group_type, "are structurally related")


def build_function_output_sections(functions: List[Dict]) -> str:
    """Build the per-function output format instructions for the FIM prompt."""
    sections = []
    for i, fn in enumerate(functions, 1):
        name = fn.get('func_name', f'function_{i}')
        sections.append(
            f"### Function {i}: `{name}`\n"
            f"```python\n<The complete function body for {name}>\n```"
        )
    return "\n\n".join(sections)


def build_function_list(functions: List[Dict]) -> str:
    """Build the function list for the FIM prompt."""
    lines = []
    for i, fn in enumerate(functions, 1):
        name = fn.get('func_name', '')
        loc = fn.get('loc', '?')
        lines.append(f"{i}. `{name}` (approx. {loc} lines)")
    return "\n".join(lines)


def build_per_function_critique_sections(functions: List[Dict], completions: Dict[str, str]) -> str:
    """Build per-function ground truth vs completion sections for the critique prompt."""
    sections = []
    for i, fn in enumerate(functions, 1):
        name = fn.get('func_name', '')
        ground_truth = fn.get('func_content', '')
        completion = completions.get(name, '<NO COMPLETION EXTRACTED>')
        sections.append(
            f"### Function {i}: `{name}`\n\n"
            f"**Ground Truth:**\n```python\n{ground_truth}\n```\n\n"
            f"**Model's Completion:**\n```python\n{completion}\n```"
        )
    return "\n\n".join(sections)



# =============================================================================
# Response Parsing
# =============================================================================

def parse_multi_fim_response(response: str, function_names: List[str]) -> dict:
    """
    Parse a multi-function FIM response.

    Expected format has multiple code blocks, one per function, each preceded
    by a heading like "### Function 1: `func_name`".

    Fallback: if only one code block is found for multiple functions, tries
    splitting by function signature patterns.
    """
    result = {
        'raw_response': response,
        'reasoning': '',
        'implementations': {},  # func_name -> implementation string
        'parse_success': False,
    }
    if not response:
        return result

    # Extract reasoning
    reasoning_match = re.search(
        r'###?\s*Reasoning\s*\n(.*?)(?=###?\s*Function\s+\d|$)',
        response, re.DOTALL | re.IGNORECASE
    )
    if reasoning_match:
        result['reasoning'] = reasoning_match.group(1).strip()

    # Strategy 1: look for named function sections
    for fname in function_names:
        short_name = fname.split('.')[-1]
        # Match "### Function N: `ClassName.method`" or "### Function N: `method`"
        pattern = (
            rf'###?\s*Function\s+\d+\s*:\s*`?{re.escape(fname)}`?\s*\n'
            rf'.*?```python\s*\n(.*?)\n```'
        )
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if not match:
            # Try short name
            pattern = (
                rf'###?\s*Function\s+\d+\s*:\s*`?{re.escape(short_name)}`?\s*\n'
                rf'.*?```python\s*\n(.*?)\n```'
            )
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            result['implementations'][fname] = match.group(1).strip()

    # Strategy 2: if named sections didn't work, try ordered code blocks
    if len(result['implementations']) < len(function_names):
        code_blocks = re.findall(r'```python\s*\n(.*?)\n```', response, re.DOTALL)
        # Skip the first block if it looks like it might be the reasoning/context
        # (heuristic: if there are more blocks than functions, skip the first)
        if len(code_blocks) > len(function_names):
            code_blocks = code_blocks[-len(function_names):]

        if len(code_blocks) == len(function_names):
            for fname, block in zip(function_names, code_blocks):
                if fname not in result['implementations']:
                    result['implementations'][fname] = block.strip()

    # Check if we got all functions
    if len(result['implementations']) == len(function_names):
        result['parse_success'] = True
    elif len(result['implementations']) > 0:
        # Partial success — log warning but still usable
        logger.warning(
            f"Partial parse: got {len(result['implementations'])}/{len(function_names)} "
            f"functions. Missing: "
            f"{set(function_names) - set(result['implementations'].keys())}"
        )
        result['parse_success'] = True  # partial is still usable

    return result


def parse_multi_critique_response(response: str) -> dict:
    """Parse the multi-function critique response."""
    result = {
        'raw_response': response,
        'parse_success': False,
        'per_function': [],
        'group_coherence': {},
        'group_overall_score': 0,
        'group_overall_reason': '',
        'should_discard': False,
        'discard_reason': '',
    }
    if not response:
        return result

    # Extract JSON
    json_match = re.search(r'```json\s*\n(.*?)\n```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = extract_json_object(response)

    if not json_str:
        logger.warning("Could not extract JSON from multi-critique response")
        return result

    try:
        parsed = json.loads(json_str)
        result['per_function'] = parsed.get('per_function', [])
        result['group_coherence'] = parsed.get('group_coherence', {})
        result['group_overall_score'] = parsed.get('group_overall_score', 0)
        result['group_overall_reason'] = parsed.get('group_overall_reason', '')
        result['should_discard'] = parsed.get('should_discard', False)
        result['discard_reason'] = parsed.get('discard_reason', '')
        result['parse_success'] = True
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse multi-critique JSON: {e}")

    return result


def extract_json_object(text: str) -> Optional[str]:
    """Extract a complete JSON object by matching balanced braces."""
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    brace_count = 0
    in_string = False
    escape_next = False
    for i, char in enumerate(text[start_idx:], start=start_idx):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start_idx:i + 1]
    return None


# =============================================================================
# Pre-Sharding
# =============================================================================

def pre_shard_input(input_path: str, total_shards: int):
    """
    Split the large input JSON into per-shard files on disk.

    Creates files like: input_shard_1_of_50.json, input_shard_2_of_50.json, ...
    Each shard file contains only the records for that shard.

    This is run ONCE before launching parallel workers, so each worker
    only reads its own small shard file (avoids OOM).
    """
    input_p = Path(input_path)
    logger.info(f"📂 Pre-sharding: loading {input_p}")
    with open(input_p, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    logger.info(f"   Total records: {total}, splitting into {total_shards} shards")

    shard_dir = input_p.parent / f"{input_p.stem}_shards_{total_shards}"
    shard_dir.mkdir(exist_ok=True)

    for shard_id in range(1, total_shards + 1):
        base_size = total // total_shards
        remainder = total % total_shards
        if shard_id <= remainder:
            start = (shard_id - 1) * (base_size + 1)
            end = start + base_size + 1
        else:
            start = remainder * (base_size + 1) + (shard_id - 1 - remainder) * base_size
            end = start + base_size

        shard_data = data[start:end]
        shard_path = shard_dir / f"shard_{shard_id}_of_{total_shards}.json"
        with open(shard_path, 'w', encoding='utf-8') as f:
            json.dump(shard_data, f, ensure_ascii=False)
        logger.info(f"   Shard {shard_id}: [{start}, {end}) → {len(shard_data)} records → {shard_path}")

    logger.info(f"✅ Pre-sharding complete. Shard directory: {shard_dir}")
    return str(shard_dir)


def get_shard_file_path(input_path: str, shard_id: int, total_shards: int) -> Path:
    """Get the path to a pre-sharded file."""
    input_p = Path(input_path)
    shard_dir = input_p.parent / f"{input_p.stem}_shards_{total_shards}"
    return shard_dir / f"shard_{shard_id}_of_{total_shards}.json"


# =============================================================================
# Main Pipeline
# =============================================================================

class MultiFIMCompletionPipeline:
    """Pipeline for multi-function FIM completion and critique."""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        checkpoint_path: str,
        model: str = "gemini-3-flash-preview",
        fim_temperature: float = 0.7,
        critique_temperature: float = 0.3,
        wait_seconds: float = 0.5,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        price_in: float = 0.50,
        price_out: float = 3.00,
        concurrency: int = 1,
        print_response: bool = False,
        use_wandb: bool = False,
        wandb_project: str = "fim-dataset-curation",
        wandb_entity: str = "",
        wandb_run_name: str = None,
        shard_id: int = None,
        total_shards: int = None,
        start_idx: int = None,
        end_idx: int = None,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.model = model
        self.fim_temperature = fim_temperature
        self.critique_temperature = critique_temperature
        self.wait_seconds = wait_seconds
        self.concurrency = max(1, concurrency)
        self.print_response = print_response

        # Sharding
        self.shard_id = shard_id
        self.total_shards = total_shards
        self.start_idx = start_idx
        self.end_idx = end_idx

        # Token counter
        self.token_counter = TokenCounter(price_in, price_out)

        # W&B
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name

        # Adjust paths for sharding
        self._adjust_paths_for_sharding()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Init wandb
        if self.use_wandb:
            self.use_wandb = self._init_wandb()

        # Init Gemini
        self.gemini_client = GeminiClient(
            model=model,
            token_counter=self.token_counter,
            use_wandb=self.use_wandb,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        # Checkpoint. `_state_lock` guards them when concurrency > 1.
        self.processed_ids = set()
        self.results_pair = []      # group_size == 2
        self.results_triple = []    # group_size == 3
        self._state_lock = threading.Lock()
        self._load_checkpoint()

    # -----------------------------------------------------------------
    # Path management
    # -----------------------------------------------------------------

    def _adjust_paths_for_sharding(self):
        shard_suffix = None
        if self.shard_id is not None and self.total_shards is not None:
            shard_suffix = f"_shard{self.shard_id}_of_{self.total_shards}"
        elif self.start_idx is not None or self.end_idx is not None:
            start = self.start_idx if self.start_idx is not None else 0
            end = self.end_idx if self.end_idx is not None else "end"
            shard_suffix = f"_range_{start}_{end}"

        if shard_suffix:
            for attr in ['output_path', 'checkpoint_path']:
                p = getattr(self, attr)
                new_p = p.parent / f"{p.stem}{shard_suffix}{p.suffix}"
                setattr(self, attr, new_p)
            logger.info(f"📂 Adjusted output: {self.output_path}")
            logger.info(f"   Adjusted checkpoint: {self.checkpoint_path}")

    def _get_output_path(self, group_size: int) -> Path:
        """Get separate output paths for pair vs triple results."""
        stem = self.output_path.stem
        suffix = self.output_path.suffix
        tag = "pairs" if group_size == 2 else "triples"
        return self.output_path.parent / f"{stem}_{tag}{suffix}"

    # -----------------------------------------------------------------
    # Sharding
    # -----------------------------------------------------------------

    def _load_shard_data(self) -> List[Dict]:
        """
        Load data for this shard. Uses pre-sharded files if available,
        otherwise falls back to loading full file + slicing.
        """
        if self.shard_id is not None and self.total_shards is not None:
            shard_file = get_shard_file_path(
                str(self.input_path), self.shard_id, self.total_shards
            )
            if shard_file.exists():
                logger.info(f"📂 Loading pre-sharded file: {shard_file}")
                with open(shard_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
                # No need to slice — file already contains only this shard's data
            else:
                logger.warning(
                    f"Pre-shard file not found: {shard_file}. "
                    f"Falling back to full-file loading. "
                    f"Run --pre-shard first to avoid OOM."
                )

        # Fallback: load full file
        logger.info(f"📂 Loading full input: {self.input_path}")
        with open(self.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total = len(data)
        start, end = self._get_shard_range(total)
        logger.info(f"   Slicing [{start}, {end}) out of {total}")
        return data[start:end]

    def _get_shard_range(self, total: int) -> tuple:
        if self.start_idx is not None or self.end_idx is not None:
            start = max(0, min(self.start_idx or 0, total))
            end = max(start, min(self.end_idx or total, total))
            return start, end
        if self.shard_id is not None and self.total_shards is not None:
            base = total // self.total_shards
            rem = total % self.total_shards
            if self.shard_id <= rem:
                start = (self.shard_id - 1) * (base + 1)
                end = start + base + 1
            else:
                start = rem * (base + 1) + (self.shard_id - 1 - rem) * base
                end = start + base
            return start, end
        return 0, total

    # -----------------------------------------------------------------
    # W&B
    # -----------------------------------------------------------------

    def _init_wandb(self) -> bool:
        base_name = self.wandb_run_name or f"multi-fim-{time.strftime('%Y%m%d-%H%M%S')}"
        if self.shard_id is not None:
            run_name = f"{base_name}-shard{self.shard_id}"
        elif self.start_idx is not None:
            run_name = f"{base_name}-range{self.start_idx}-{self.end_idx or 'end'}"
        else:
            run_name = base_name
        return init_wandb(
            project=self.wandb_project,
            run_name=run_name,
            entity=self.wandb_entity,
            config={
                "model": self.model,
                "input_path": str(self.input_path),
                "concurrency": self.concurrency,
                "shard_id": self.shard_id,
                "total_shards": self.total_shards,
            },
        )

    # -----------------------------------------------------------------
    # Checkpoint
    # -----------------------------------------------------------------

    def _load_checkpoint(self):
        if not self.checkpoint_path.exists():
            return
        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                ckpt = json.load(f)
            self.processed_ids = set(ckpt.get('processed_ids', []))
            self.results_pair = ckpt.get('results_pair', [])
            self.results_triple = ckpt.get('results_triple', [])
            if 'token_usage' in ckpt:
                self.token_counter.load_from_dict(ckpt['token_usage'])
            logger.info(
                f"📂 Checkpoint loaded: {len(self.processed_ids)} processed, "
                f"{len(self.results_pair)} pairs, {len(self.results_triple)} triples"
            )
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    def _save_checkpoint(self):
        ckpt = {
            'processed_ids': list(self.processed_ids),
            'results_pair': self.results_pair,
            'results_triple': self.results_triple,
            'token_usage': self.token_counter.to_dict(),
        }
        temp = self.checkpoint_path.with_suffix('.tmp')
        try:
            with open(temp, 'w', encoding='utf-8') as f:
                json.dump(ckpt, f, ensure_ascii=False, indent=2)
            temp.replace(self.checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp.exists():
                temp.unlink()

    # -----------------------------------------------------------------
    # Debug helpers
    # -----------------------------------------------------------------

    def _print_response(self, title: str, uid: str, response: str, usage: dict):
        print("\n" + "=" * 80)
        print(f"📝 {title} for: {uid}")
        print(f"   Tokens: in={usage['input_tokens']:,}, out={usage['output_tokens']:,}, "
              f"latency={usage.get('latency', 0):.2f}s")
        print("=" * 80)
        print(response[:3000] + "..." if len(response) > 3000 else response)
        print("=" * 80 + "\n")

    def _print_current_cost(self):
        cost = self.token_counter.get_cost()
        print(f"💰 Cumulative: ${cost['total_cost']:.4f} "
              f"(in:{cost['input_tokens']:,} out:{cost['output_tokens']:,} "
              f"FIM:{cost['fim_requests']} Critique:{cost['critique_requests']})")

    # -----------------------------------------------------------------
    # Core: process one group record
    # -----------------------------------------------------------------

    def process_single_record(self, record: dict) -> Optional[dict]:
        """
        Process one group record: multi-FIM completion + multi-critique.
        """
        unique_id = make_unique_id(record)
        functions = record.get('functions', [])
        function_names = [f['func_name'] for f in functions]
        masked_code = record.get('masked_code_content', '')
        group_type = record.get('group_type', '')
        group_size = record.get('group_size', len(functions))

        # Copy all original fields
        result = record.copy()
        result['unique_id'] = unique_id
        # Drop heavy masked_code from output (it can be regenerated)
        # Keep it for now, remove at final save if needed

        relationship_desc = get_relationship_description(group_type)

        # =================================================================
        # Step 1: Multi-FIM Completion
        # =================================================================
        fim_prompt = MULTI_FIM_COMPLETION_PROMPT.format(
            num_functions=group_size,
            relationship_description=relationship_desc,
            function_output_sections=build_function_output_sections(functions),
            masked_code=masked_code,
            function_list=build_function_list(functions),
        )

        try:
            fim_response, fim_usage = self.gemini_client.get_response(
                prompt=fim_prompt,
                sample_id=unique_id,
                call_type="fim",
                temperature=self.fim_temperature,
            )
            if self.print_response:
                self._print_response("Multi-FIM Completion", unique_id, fim_response, fim_usage)

            fim_parsed = parse_multi_fim_response(fim_response, function_names)

            result['fim_response'] = {
                'raw_response': fim_response,
                'reasoning': fim_parsed['reasoning'],
                'implementations': fim_parsed['implementations'],
                'parse_success': fim_parsed['parse_success'],
                'token_usage': fim_usage,
            }

            if self.wait_seconds > 0:
                time.sleep(self.wait_seconds)

        except Exception as e:
            logger.error(f"Multi-FIM failed for {unique_id}: {e}")
            result['fim_response'] = {'error': str(e), 'parse_success': False}
            result['should_discard'] = None
            result['discard_reason'] = f"FIM API call failed: {e}"
            return result

        # =================================================================
        # Step 2: Multi-Critique
        # =================================================================
        completions = fim_parsed['implementations'] if fim_parsed['parse_success'] else {}

        critique_prompt = MULTI_CRITIQUE_PROMPT.format(
            num_functions=group_size,
            relationship_description=relationship_desc,
            masked_code=masked_code,
            per_function_sections=build_per_function_critique_sections(functions, completions),
        )

        try:
            critique_response, critique_usage = self.gemini_client.get_response(
                prompt=critique_prompt,
                sample_id=unique_id,
                call_type="critique",
                temperature=self.critique_temperature,
            )
            if self.print_response:
                self._print_response("Multi-Critique", unique_id, critique_response, critique_usage)

            critique_parsed = parse_multi_critique_response(critique_response)

            result['critique_response'] = {
                'raw_response': critique_response,
                'per_function': critique_parsed['per_function'],
                'group_coherence': critique_parsed['group_coherence'],
                'group_overall_score': critique_parsed['group_overall_score'],
                'group_overall_reason': critique_parsed['group_overall_reason'],
                'should_discard': critique_parsed['should_discard'],
                'discard_reason': critique_parsed['discard_reason'],
                'parse_success': critique_parsed['parse_success'],
                'token_usage': critique_usage,
            }

            # Promote to top-level
            result['should_discard'] = critique_parsed['should_discard']
            result['discard_reason'] = critique_parsed['discard_reason']
            result['group_overall_score'] = critique_parsed['group_overall_score']
            result['group_coherence_score'] = critique_parsed.get(
                'group_coherence', {}
            ).get('coherence_score', 0)

            # Compute per-function feasibility summary
            per_fn = critique_parsed.get('per_function', [])
            result['all_feasible'] = all(
                f.get('feasibility', {}).get('is_feasible', True)
                for f in per_fn
            ) if per_fn else None

            if self.wait_seconds > 0:
                time.sleep(self.wait_seconds)

        except Exception as e:
            logger.error(f"Multi-critique failed for {unique_id}: {e}")
            result['critique_response'] = {'error': str(e), 'parse_success': False}
            result['should_discard'] = None
            result['discard_reason'] = f"Critique API call failed: {e}"

        # W&B logging
        if self.use_wandb and result.get('critique_response', {}).get('parse_success'):
            cr = result['critique_response']
            log_data = {
                "scores/group_overall": cr['group_overall_score'],
                "coherence/score": cr.get('group_coherence', {}).get('coherence_score', 0),
                "decision/should_discard": cr['should_discard'],
                "meta/group_size": group_size,
                "meta/group_type": group_type,
            }
            for pf in cr.get('per_function', []):
                fn = pf.get('func_name', 'unknown')
                log_data[f"per_func/{fn}/overall"] = pf.get('overall_score', 0)
            wandb.log(log_data)

        return result

    # -----------------------------------------------------------------
    # Run pipeline
    # -----------------------------------------------------------------

    def _process_and_record(self, record: dict, pbar, shard_prefix: str):
        """
        Process one group, then commit it to the checkpoint.

        Pairs and triples are accumulated separately so they can be written to
        separate output files. A failed group is still marked processed so a
        restart doesn't retry it forever.
        """
        uid = record['_unique_id']
        cost = self.token_counter.get_cost()
        pbar.set_description(
            f"{shard_prefix}Done:{len(self.processed_ids)} "
            f"Cost:${cost['total_cost']:.3f}"
        )

        try:
            record_clean = {k: v for k, v in record.items() if k != '_unique_id'}
            result = self.process_single_record(record_clean)

            if result:
                gs = result.get('group_size', 2)
                with self._state_lock:
                    if gs == 3:
                        self.results_triple.append(result)
                    else:
                        self.results_pair.append(result)
                    self.processed_ids.add(uid)

                score = result.get('group_overall_score', 'N/A')
                coh = result.get('group_coherence_score', 'N/A')
                disc = result.get('should_discard', 'N/A')
                logger.info(
                    f"  ✅ {uid}: score={score}, coherence={coh}, discard={disc}"
                )

        except Exception as e:
            logger.error(f"Error processing {uid}: {e}")
            with self._state_lock:
                self.processed_ids.add(uid)

        with self._state_lock:
            self._save_checkpoint()
        self._print_current_cost()
        pbar.update(1)

    def run(self):
        """Run the multi-function FIM pipeline."""
        data = self._load_shard_data()
        logger.info(f"📊 Records in this shard: {len(data)}")

        # Assign unique IDs
        for record in data:
            record['_unique_id'] = make_unique_id(record)

        remaining = [r for r in data if r['_unique_id'] not in self.processed_ids]
        logger.info(f"   Already processed: {len(self.processed_ids)}")
        logger.info(f"   Remaining: {len(remaining)}")
        self._print_current_cost()

        shard_prefix = f"[Shard {self.shard_id}/{self.total_shards}] " if self.shard_id else ""
        if self.concurrency > 1:
            logger.info(f"⚡ In-process concurrency: {self.concurrency} requests in flight")

        with tqdm(total=len(remaining), desc=f"{shard_prefix}Processing") as pbar:
            if self.concurrency == 1:
                for record in remaining:
                    self._process_and_record(record, pbar, shard_prefix)
            else:
                with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                    futures = [
                        pool.submit(self._process_and_record, record, pbar, shard_prefix)
                        for record in remaining
                    ]
                    for fut in as_completed(futures):
                        fut.result()

        # Save final outputs (separate files for pairs and triples)
        for group_size, results, tag in [
            (2, self.results_pair, "pairs"),
            (3, self.results_triple, "triples"),
        ]:
            if results:
                out_path = self._get_output_path(group_size)
                logger.info(f"💾 Saving {len(results)} {tag} → {out_path}")
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        self._print_summary()

        if self.use_wandb:
            finish_wandb()

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    def _print_summary(self):
        all_results = self.results_pair + self.results_triple
        print("\n" + "=" * 60)
        print("📊 Multi-Function FIM Processing Summary")
        if self.shard_id:
            print(f"   Shard: {self.shard_id}/{self.total_shards}")
        print("=" * 60)

        print(f"\n📈 Counts:")
        print(f"   Total processed: {len(self.processed_ids)}")
        print(f"   Pair results:    {len(self.results_pair)}")
        print(f"   Triple results:  {len(self.results_triple)}")

        self.token_counter.print_cost_summary()

        if all_results:
            # Discard stats
            keep = sum(1 for r in all_results if r.get('should_discard') is False)
            discard = sum(1 for r in all_results if r.get('should_discard') is True)
            unknown = sum(1 for r in all_results if r.get('should_discard') is None)
            print(f"\n🗑️ Discard Decision:")
            print(f"   Keep: {keep}  Discard: {discard}  Unknown: {unknown}")

            # Group type breakdown
            type_counts = {}
            for r in all_results:
                gt = r.get('group_type', '?')
                type_counts[gt] = type_counts.get(gt, 0) + 1
            print(f"\n📋 Group Type Breakdown:")
            for gt, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
                print(f"   {gt}: {cnt}")

            # Score distribution
            scores = [
                r.get('group_overall_score', 0)
                for r in all_results
                if r.get('critique_response', {}).get('parse_success')
            ]
            if scores:
                avg = sum(scores) / len(scores)
                print(f"\n📊 Group Overall Score:")
                print(f"   Average: {avg:.2f}")
                dist = {}
                for s in scores:
                    dist[s] = dist.get(s, 0) + 1
                for s in sorted(dist):
                    print(f"   Score {s}: {dist[s]} ({dist[s]/len(scores)*100:.1f}%)")

            # Coherence scores
            coh_scores = [
                r.get('group_coherence_score', 0)
                for r in all_results
                if r.get('group_coherence_score') is not None
                and r.get('group_coherence_score') > 0
            ]
            if coh_scores:
                print(f"\n🔗 Group Coherence Score:")
                print(f"   Average: {sum(coh_scores)/len(coh_scores):.2f}")

            # Parse success
            fim_ok = sum(1 for r in all_results if r.get('fim_response', {}).get('parse_success'))
            cri_ok = sum(1 for r in all_results if r.get('critique_response', {}).get('parse_success'))
            n = len(all_results)
            print(f"\n📋 Parse Success Rate:")
            print(f"   FIM: {fim_ok}/{n} ({fim_ok/max(n,1)*100:.1f}%)")
            print(f"   Critique: {cri_ok}/{n} ({cri_ok/max(n,1)*100:.1f}%)")

        print("\n" + "=" * 60)




# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-function FIM completion and critique (step 4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Step 0 — pre-split the input once, so parallel workers don't each load it all:
  python multi_function/step_4_multi_fim_and_critique.py --pre-shard --total-shards 200

  # Step 1 — run one shard (scripts/run_step4_parallel.sh fans out the rest):
  python multi_function/step_4_multi_fim_and_critique.py --shard 1 --total-shards 200

  # Single process over everything:
  python multi_function/step_4_multi_fim_and_critique.py

  # Smoke-test 20 groups before spending real money:
  python multi_function/step_4_multi_fim_and_critique.py --start-idx 0 --end-idx 20 --print-response
        '''
    )

    add_config_arg(parser)

    # Special mode: pre-shard only
    parser.add_argument("--pre-shard", action="store_true",
                        help="Pre-split the input into per-shard files on disk, then exit")

    # I/O
    parser.add_argument("--input", "-i", default=None,
                        help="Override step 3's *_groups.json")
    parser.add_argument("--output", "-o", default=None,
                        help="Override <work_dir>/multi_function/step_4_multi_fim.json")
    parser.add_argument("--checkpoint", "-c", default=None,
                        help="Checkpoint path (default: <output stem>_checkpoint.json)")

    # Model
    parser.add_argument("--model", "-m", default=None, help="Override llm.model")
    parser.add_argument("--fim-temperature", type=float, default=None)
    parser.add_argument("--critique-temperature", type=float, default=None)

    # Processing
    parser.add_argument("--wait", "-w", type=float, default=None,
                        help="Seconds to sleep after each API call (override llm.wait_seconds)")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Requests in flight within this process (override sharding.concurrency)")
    parser.add_argument("--print-response", "-p", action="store_true",
                        help="Print every API response — use on a small slice only")

    # Sharding
    parser.add_argument("--shard", type=int, default=None, help="Shard ID (1-indexed)")
    parser.add_argument("--total-shards", type=int, default=None)
    parser.add_argument("--start-idx", type=int, default=None)
    parser.add_argument("--end-idx", type=int, default=None)

    # W&B
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)

    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = derive_paths(cfg)
    llm = cfg["llm"]

    input_path = Path(args.input) if args.input else paths["multi_step3_groups"]
    output_path = Path(args.output) if args.output else paths["multi_step4_out"]

    if not input_path.exists():
        raise SystemExit(
            f"Error: input not found: {input_path}\n"
            "Run multi_function/step_3_select_function_groups.py first."
        )

    # --pre-shard mode: split and exit.
    if args.pre_shard:
        total_shards = args.total_shards or cfg["sharding"]["total_shards"]
        pre_shard_input(str(input_path), total_shards)
        return

    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        output_path.parent / f"{output_path.stem}_checkpoint.json"
    )

    # Validate sharding args
    if args.shard is not None and args.total_shards is None:
        parser.error("--shard requires --total-shards")
    if args.total_shards is not None and args.shard is None:
        parser.error("--total-shards requires --shard (or use --pre-shard)")
    if args.shard is not None and (args.start_idx is not None or args.end_idx is not None):
        parser.error("Cannot use --shard with --start-idx / --end-idx")

    use_wandb = args.wandb or cfg["wandb"]["enabled"]
    if use_wandb and not WANDB_AVAILABLE:
        logger.warning("⚠️ W&B requested but not installed — continuing without it")
        use_wandb = False

    pipeline = MultiFIMCompletionPipeline(
        input_path=str(input_path),
        output_path=str(output_path),
        checkpoint_path=str(checkpoint),
        model=args.model or llm["model"],
        fim_temperature=args.fim_temperature if args.fim_temperature is not None else llm["fim_temperature"],
        critique_temperature=args.critique_temperature if args.critique_temperature is not None else llm["critique_temperature"],
        wait_seconds=args.wait if args.wait is not None else llm["wait_seconds"],
        max_retries=llm["max_retries"],
        retry_delay=llm["retry_delay"],
        price_in=llm["price_per_1m_input_tokens"],
        price_out=llm["price_per_1m_output_tokens"],
        concurrency=args.concurrency if args.concurrency is not None else cfg["sharding"]["concurrency"],
        print_response=args.print_response,
        use_wandb=use_wandb,
        wandb_project=args.wandb_project or cfg["wandb"]["project"],
        wandb_entity=cfg["wandb"]["entity"],
        wandb_run_name=args.wandb_run_name or cfg["wandb"]["run_name"] or None,
        shard_id=args.shard,
        total_shards=args.total_shards,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
