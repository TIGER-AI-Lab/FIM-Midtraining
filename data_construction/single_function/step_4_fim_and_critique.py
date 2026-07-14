#!/usr/bin/env python3
"""
Step 4 (single-function) — generate the completion and its critique.

For every masked function from step 3:
  1. Ask Gemini to complete the masked body, reasoning step by step first.
     The reasoning IS the training signal — it becomes the assistant turn.
  2. Ask Gemini to critique that completion against the ground truth, and to
     judge whether the function was *fair game* in the first place (a body that
     depends on an obscure external API can't be inferred from context, no
     matter how good the model is — those get discarded rather than counted
     as failures).

This is the only step that costs money. It is shardable and checkpointed:
each shard owns its own checkpoint file, saves after every record, and skips
what it has already done on restart. Killing and relaunching a shard is safe.

    # single process, whole dataset
    python single_function/step_4_fim_and_critique.py

    # one shard of 200 (see scripts/run_step4_parallel.sh for the full fan-out)
    python single_function/step_4_fim_and_critique.py --shard 1 --total-shards 200
"""

import json
import logging
import re
import threading
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import sys

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
# Unique ID Generation (single source of truth)
# =============================================================================

def make_unique_id(record: dict) -> str:
    """
    Generate a deterministic unique ID for a record.

    Uses sample_id + func_name + start_line to ensure uniqueness.
    This is a module-level function so that run() and process_single_record()
    always produce the same ID for the same record — critical for checkpoint
    consistency.
    """
    sample_id = record.get('sample_id', '')
    func_name = record.get('func_name', '')
    start_line = record.get('start_line', 0)
    return f"{sample_id}_{func_name}_{start_line}"


# =============================================================================
# Prompt Templates
# =============================================================================

FIM_COMPLETION_PROMPT = '''You are an expert Python programmer. Your task is to complete a masked function based on the surrounding code context.

## Task

Below is a Python file where one function's body has been replaced with `# <MASKED_FUNCTION_BODY>`. Your job is to:

1. **Analyze the context**: Look at the imports, other functions, class definitions, and how this function is called or calls other functions.
2. **Reason step by step**: Think through what this function should do based on:
   - The function signature (name, parameters, type hints)
   - The docstring (if visible in the signature area)
   - How it's used elsewhere in the code
   - What other functions it might need to call
   - The overall purpose of the code
3. **Write the complete function body**: Provide a working implementation.

## Output Format

Please structure your response as follows:

### Reasoning
<Your step-by-step analysis of what the function should do>

### Implementation
```python
<The complete function body code only, without the function signature>
```

## Important Notes
- Only provide the function BODY (the code inside the function), not the signature
- Maintain proper indentation (the body should be indented with 4 spaces or appropriate level)
- The implementation should be consistent with the coding style in the file
- If the function has a docstring, include it as part of the body

## Code with Masked Function

```python
{masked_code}
```

## Function to Complete

Function name: `{function_name}`

Please analyze the context and provide your reasoning followed by the implementation.
'''

CRITIQUE_PROMPT = '''You are an expert code reviewer. Your task is to evaluate a function completion against the ground truth implementation, AND determine whether this function is suitable for FIM (Fill-in-the-Middle) evaluation.

## Task

A model was asked to complete a masked function based on surrounding code context. You need to:
1. **Feasibility Check**: First, determine whether this function is actually feasible to complete based solely on the code context provided. Some functions rely on obscure external libraries, domain-specific APIs, or conventions that cannot be reasonably inferred from the surrounding code. These functions should be marked as infeasible.
2. **Quality Evaluation**: Compare the completion with the ground truth and score the completion across multiple dimensions.

## Context (Code with Masked Function)

```python
{masked_code}
```

## Function Being Evaluated

Function name: `{function_name}`

## Ground Truth Implementation

```python
{ground_truth}
```

## Model's Completion

```python
{completion}
```

## Part 1: Feasibility Assessment

Determine whether this function is a FAIR test for FIM completion. A function is **infeasible** if:
- It relies heavily on **uncommon or domain-specific external libraries/APIs** whose usage patterns cannot be inferred from the visible context (e.g., calling specific methods of a niche library that isn't used elsewhere in the file).
- It requires **knowledge of external conventions, configurations, or data formats** not present in the file.
- The surrounding context provides **insufficient information** to reconstruct the core logic (e.g., the function implements an algorithm whose details are not hinted at anywhere in the file).
- It uses **magic constants, hardcoded strings, or lookup tables** that are specific to external systems and cannot be guessed.

A function is **feasible** if:
- Its behavior can be reasonably inferred from the function name, signature, docstring, and surrounding code.
- It follows patterns established by other functions in the file.
- It uses standard Python or well-known library APIs that are already imported/used in the file.
- Even if the exact implementation differs from ground truth, a competent programmer could write a functionally equivalent version from the available context.

## Part 2: Quality Evaluation Criteria

Evaluate the completion on these dimensions (1-5 scale, where 5 is best):

1. **Correctness (1-5)**: Does the implementation produce the same results as the ground truth for all inputs?
   - 5: Functionally identical
   - 4: Correct for most cases, minor edge case differences
   - 3: Partially correct, some important cases wrong
   - 2: Major functional differences
   - 1: Completely incorrect

2. **Executability (1-5)**: Can the code actually run without errors? Check for syntax errors, indentation issues, undefined variables, missing imports, type errors, etc.
   - 5: Code is fully executable, no errors
   - 4: Minor issues that are easy to fix (e.g., a small typo, slightly off indentation)
   - 3: Some issues but the main logic is executable (e.g., missing an edge case handler that would crash)
   - 2: Significant issues that would prevent execution (e.g., syntax errors, undefined variables)
   - 1: Code is pseudo-code or completely non-executable (e.g., placeholder comments, wrong language syntax)

3. **API Usage (1-5)**: Does it correctly use other functions, methods, and external APIs present in the codebase?
   - 5: All calls match ground truth
   - 4: Minor differences in API usage
   - 3: Some incorrect or missing API calls
   - 2: Major API usage errors
   - 1: Completely wrong API usage

4. **Readability (1-5)**: Is the code clean, well-formatted, and follows Python conventions?
   - 5: Excellent readability, possibly better than ground truth
   - 4: Good readability, comparable to ground truth
   - 3: Acceptable readability
   - 2: Poor readability
   - 1: Very hard to read

5. **Completeness (1-5)**: Does the implementation handle all cases the ground truth handles?
   - 5: Handles all cases including edge cases
   - 4: Handles main cases, may miss some edge cases
   - 3: Handles basic cases only
   - 2: Missing important cases
   - 1: Very incomplete

## Output Format

Please provide your analysis in the following JSON format:

```json
{{
  "feasibility": {{
    "is_feasible": <true or false>,
    "confidence": <0.0 to 1.0, how confident you are in this judgment>,
    "reason": "<detailed explanation of why this function is or isn't feasible for FIM evaluation>",
    "infeasible_factors": ["<list of specific factors making it infeasible, empty if feasible>"]
  }},
  "reasoning": "<Your detailed step-by-step analysis comparing the two implementations>",
  "scores": {{
    "correctness": <1-5>,
    "correctness_reason": "<brief explanation>",
    "executability": <1-5>,
    "executability_reason": "<brief explanation, list specific issues if any>",
    "api_usage": <1-5>,
    "api_usage_reason": "<brief explanation>",
    "readability": <1-5>,
    "readability_reason": "<brief explanation>",
    "completeness": <1-5>,
    "completeness_reason": "<brief explanation>"
  }},
  "overall_score": <1-5>,
  "overall_reason": "<summary of the overall quality>",
  "key_differences": ["<list of main differences>"],
  "key_similarities": ["<list of main similarities>"],
  "should_discard": <true or false>,
  "discard_reason": "<if should_discard is true, explain why this sample should be discarded from the dataset>"
}}
```

## Decision Logic for `should_discard`

Set `should_discard` to `true` if ANY of the following conditions are met:
1. **Infeasible**: `is_feasible` is `false` — the function fundamentally cannot be completed from context alone.
2. **Very Low Quality**: `overall_score` <= 2 AND `executability` <= 2 — the model's completion is too poor to be useful as training/evaluation data.
3. **Non-executable Garbage**: `executability` == 1 — the completion is completely non-functional.

Otherwise, set `should_discard` to `false`. Even if the completion isn't perfect, it can still be valuable data if the function is feasible and the completion is at least partially correct or executable.

## Scoring Guidelines for Overall Score
- **5**: The completion is functionally equivalent to the ground truth. It would work correctly as a drop-in replacement.
- **4**: The completion is mostly correct with minor issues that wouldn't significantly impact functionality.
- **3**: The completion captures the main idea but has notable differences or issues.
- **2**: The completion has major problems but shows some understanding of the task.
- **1**: The completion is fundamentally incorrect or completely misses the point.

Note: A high correctness score but low executability score indicates the model understood the logic but made implementation mistakes. This distinction is important for training data quality assessment.

Please provide your detailed analysis and scores.
'''


# =============================================================================
# Response Parsing
# =============================================================================

def parse_fim_response(response: str) -> dict:
    """Parse the FIM completion response to extract reasoning and implementation."""
    result = {
        'raw_response': response,
        'reasoning': '',
        'implementation': '',
        'parse_success': False
    }

    if not response:
        return result

    # Try to extract reasoning section
    reasoning_match = re.search(
        r'###?\s*Reasoning\s*\n(.*?)(?=###?\s*Implementation|```python)',
        response,
        re.DOTALL | re.IGNORECASE
    )
    if reasoning_match:
        result['reasoning'] = reasoning_match.group(1).strip()

    # Try to extract implementation from code block
    code_match = re.search(
        r'```python\s*\n(.*?)\n```',
        response,
        re.DOTALL
    )
    if code_match:
        result['implementation'] = code_match.group(1).strip()
        result['parse_success'] = True
    else:
        # Fallback: try to find any code after "Implementation"
        impl_match = re.search(
            r'###?\s*Implementation\s*\n(.*)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if impl_match:
            result['implementation'] = impl_match.group(1).strip()
            result['parse_success'] = True

    return result


def parse_critique_response(response: str) -> dict:
    """Parse the critique response to extract feasibility, scores, and reasoning."""
    result = {
        'raw_response': response,
        'parse_success': False,
        # Feasibility fields
        'feasibility': {
            'is_feasible': True,
            'confidence': 0.0,
            'reason': '',
            'infeasible_factors': []
        },
        # Quality fields
        'reasoning': '',
        'scores': {},
        'overall_score': 0,
        'overall_reason': '',
        'key_differences': [],
        'key_similarities': [],
        # Discard decision
        'should_discard': False,
        'discard_reason': ''
    }

    if not response:
        return result

    # Try to extract JSON from code block
    json_match = re.search(r'```json\s*\n(.*?)\n```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = extract_json_object(response)

    if not json_str:
        logger.warning("Could not extract JSON from critique response")
        return result

    try:
        parsed = json.loads(json_str)

        # Parse feasibility
        feasibility = parsed.get('feasibility', {})
        result['feasibility'] = {
            'is_feasible': feasibility.get('is_feasible', True),
            'confidence': feasibility.get('confidence', 0.0),
            'reason': feasibility.get('reason', ''),
            'infeasible_factors': feasibility.get('infeasible_factors', [])
        }

        # Parse quality scores
        result['reasoning'] = parsed.get('reasoning', '')
        result['scores'] = parsed.get('scores', {})
        result['overall_score'] = parsed.get('overall_score', 0)
        result['overall_reason'] = parsed.get('overall_reason', '')
        result['key_differences'] = parsed.get('key_differences', [])
        result['key_similarities'] = parsed.get('key_similarities', [])

        # Parse discard decision
        result['should_discard'] = parsed.get('should_discard', False)
        result['discard_reason'] = parsed.get('discard_reason', '')

        result['parse_success'] = True
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse critique JSON: {e}")

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
# Main Pipeline
# =============================================================================

class FIMCompletionPipeline:
    """Pipeline for FIM completion and critique."""

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

        # Sharding parameters
        self.shard_id = shard_id
        self.total_shards = total_shards
        self.start_idx = start_idx
        self.end_idx = end_idx

        # Token counter
        self.token_counter = TokenCounter(price_in, price_out)

        # W&B configuration
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name

        # Adjust paths for sharding (must happen before _load_checkpoint)
        self._adjust_paths_for_sharding()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize wandb if enabled
        if self.use_wandb:
            self.use_wandb = self._init_wandb()

        # Initialize Gemini client
        self.gemini_client = GeminiClient(
            model=model,
            token_counter=self.token_counter,
            use_wandb=self.use_wandb,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        # Checkpoint data. `_state_lock` guards them when concurrency > 1.
        self.processed_ids = set()
        self.results = []
        self._state_lock = threading.Lock()
        self._load_checkpoint()

    def _adjust_paths_for_sharding(self):
        """Adjust output and checkpoint paths based on sharding configuration."""
        shard_suffix = None
        if self.shard_id is not None and self.total_shards is not None:
            shard_suffix = f"_shard{self.shard_id}_of_{self.total_shards}"
        elif self.start_idx is not None or self.end_idx is not None:
            start = self.start_idx if self.start_idx is not None else 0
            end = self.end_idx if self.end_idx is not None else "end"
            shard_suffix = f"_range_{start}_{end}"

        if shard_suffix:
            output_stem = self.output_path.stem
            output_suffix = self.output_path.suffix
            self.output_path = self.output_path.parent / f"{output_stem}{shard_suffix}{output_suffix}"

            checkpoint_stem = self.checkpoint_path.stem
            checkpoint_suffix = self.checkpoint_path.suffix
            self.checkpoint_path = self.checkpoint_path.parent / f"{checkpoint_stem}{shard_suffix}{checkpoint_suffix}"

            logger.info(f"📂 Adjusted paths for sharding:")
            logger.info(f"   Output: {self.output_path}")
            logger.info(f"   Checkpoint: {self.checkpoint_path}")

    def _get_shard_range(self, total_samples: int) -> tuple[int, int]:
        """Calculate the start and end indices for this shard."""
        if self.start_idx is not None or self.end_idx is not None:
            start = self.start_idx if self.start_idx is not None else 0
            end = self.end_idx if self.end_idx is not None else total_samples
            start = max(0, min(start, total_samples))
            end = max(start, min(end, total_samples))
            return start, end

        if self.shard_id is not None and self.total_shards is not None:
            if self.shard_id < 1 or self.shard_id > self.total_shards:
                raise ValueError(f"shard_id must be between 1 and {self.total_shards}")
            base_size = total_samples // self.total_shards
            remainder = total_samples % self.total_shards
            if self.shard_id <= remainder:
                start = (self.shard_id - 1) * (base_size + 1)
                end = start + base_size + 1
            else:
                start = remainder * (base_size + 1) + (self.shard_id - 1 - remainder) * base_size
                end = start + base_size
            return start, end

        return 0, total_samples

    def _init_wandb(self) -> bool:
        """Initialize W&B and Weave. Returns False if wandb is unavailable."""
        logger.info("🔧 Initializing W&B and Weave...")
        base_run_name = self.wandb_run_name or f"fim-critique-{time.strftime('%Y%m%d-%H%M%S')}"
        if self.shard_id is not None and self.total_shards is not None:
            run_name = f"{base_run_name}-shard{self.shard_id}"
        elif self.start_idx is not None or self.end_idx is not None:
            start = self.start_idx if self.start_idx is not None else 0
            end = self.end_idx if self.end_idx is not None else "end"
            run_name = f"{base_run_name}-range{start}-{end}"
        else:
            run_name = base_run_name

        return init_wandb(
            project=self.wandb_project,
            run_name=run_name,
            entity=self.wandb_entity,
            config={
                "model": self.model,
                "input_path": str(self.input_path),
                "output_path": str(self.output_path),
                "fim_temperature": self.fim_temperature,
                "critique_temperature": self.critique_temperature,
                "wait_seconds": self.wait_seconds,
                "concurrency": self.concurrency,
                "shard_id": self.shard_id,
                "total_shards": self.total_shards,
            },
        )

    def _load_checkpoint(self):
        """Load checkpoint data if exists."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                self.processed_ids = set(checkpoint.get('processed_ids', []))
                self.results = checkpoint.get('results', [])
                if 'token_usage' in checkpoint:
                    self.token_counter.load_from_dict(checkpoint['token_usage'])
                logger.info(f"📂 Loaded checkpoint:")
                logger.info(f"   Processed records: {len(self.processed_ids)}")
                logger.info(f"   Results saved: {len(self.results)}")
                cost = self.token_counter.get_cost()
                logger.info(f"   Total cost so far: ${cost['total_cost']:.4f}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

    def _save_checkpoint(self):
        """Save current progress to checkpoint file (atomic write via temp file)."""
        checkpoint = {
            'processed_ids': list(self.processed_ids),
            'results': self.results,
            'token_usage': self.token_counter.to_dict()
        }
        temp_path = self.checkpoint_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            temp_path.replace(self.checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()

    def _print_response(self, title: str, unique_id: str, response: str, usage_info: dict):
        """Print API response for debugging."""
        print("\n" + "=" * 80)
        print(f"📝 {title} for: {unique_id}")
        print(f"   Tokens: input={usage_info['input_tokens']:,}, output={usage_info['output_tokens']:,}")
        print(f"   Latency: {usage_info.get('latency', 0):.2f}s")
        print("=" * 80)
        print(response[:2000] + "..." if len(response) > 2000 else response)
        print("=" * 80 + "\n")

    def _print_current_cost(self):
        """Print current token usage and cost."""
        cost = self.token_counter.get_cost()
        print(f"💰 Cumulative cost: ${cost['total_cost']:.4f} "
              f"(in: {cost['input_tokens']:,}, out: {cost['output_tokens']:,}, "
              f"FIM: {cost['fim_requests']}, Critique: {cost['critique_requests']})")

    def process_single_record(self, record: dict) -> Optional[dict]:
        """
        Process a single function record: FIM completion + Critique.

        Returns:
            The record with added completion, critique, and discard decision.
        """
        unique_id = make_unique_id(record)  # Module-level function — same as run()
        function_name = record.get('func_name', '')
        masked_code = record.get('masked_code_content', '')
        ground_truth = record.get('func_content', '')

        # Start with all original fields preserved
        result = record.copy()
        result['unique_id'] = unique_id

        # =====================================================================
        # Step 1: FIM Completion
        # =====================================================================
        fim_prompt = FIM_COMPLETION_PROMPT.format(
            masked_code=masked_code,
            function_name=function_name
        )

        try:
            fim_response, fim_usage = self.gemini_client.get_response(
                prompt=fim_prompt,
                sample_id=unique_id,
                call_type="fim",
                temperature=self.fim_temperature
            )

            if self.print_response:
                self._print_response("FIM Completion", unique_id, fim_response, fim_usage)

            fim_parsed = parse_fim_response(fim_response)

            result['fim_response'] = {
                'raw_response': fim_response,
                'reasoning': fim_parsed['reasoning'],
                'implementation': fim_parsed['implementation'],
                'parse_success': fim_parsed['parse_success'],
                'token_usage': fim_usage
            }

            if self.wait_seconds > 0:
                time.sleep(self.wait_seconds)

        except Exception as e:
            logger.error(f"FIM completion failed for {unique_id}: {e}")
            result['fim_response'] = {
                'error': str(e),
                'parse_success': False
            }
            result['should_discard'] = None  # Unknown — API error
            result['discard_reason'] = f"FIM API call failed: {e}"
            result['is_feasible'] = None  # Unknown — never evaluated
            return result

        # =====================================================================
        # Step 2: Critique (includes feasibility check)
        # =====================================================================
        completion = fim_parsed['implementation'] if fim_parsed['parse_success'] else fim_response

        critique_prompt = CRITIQUE_PROMPT.format(
            masked_code=masked_code,
            function_name=function_name,
            ground_truth=ground_truth,
            completion=completion
        )

        try:
            critique_response, critique_usage = self.gemini_client.get_response(
                prompt=critique_prompt,
                sample_id=unique_id,
                call_type="critique",
                temperature=self.critique_temperature
            )

            if self.print_response:
                self._print_response("Critique", unique_id, critique_response, critique_usage)

            critique_parsed = parse_critique_response(critique_response)

            result['critique_response'] = {
                'raw_response': critique_response,
                'feasibility': critique_parsed['feasibility'],
                'reasoning': critique_parsed['reasoning'],
                'scores': critique_parsed['scores'],
                'overall_score': critique_parsed['overall_score'],
                'overall_reason': critique_parsed['overall_reason'],
                'key_differences': critique_parsed['key_differences'],
                'key_similarities': critique_parsed['key_similarities'],
                'should_discard': critique_parsed['should_discard'],
                'discard_reason': critique_parsed['discard_reason'],
                'parse_success': critique_parsed['parse_success'],
                'token_usage': critique_usage
            }

            # Promote discard decision to top level for easy filtering
            result['should_discard'] = critique_parsed['should_discard']
            result['discard_reason'] = critique_parsed['discard_reason']
            result['is_feasible'] = critique_parsed['feasibility']['is_feasible']

            if self.wait_seconds > 0:
                time.sleep(self.wait_seconds)

        except Exception as e:
            logger.error(f"Critique failed for {unique_id}: {e}")
            result['critique_response'] = {
                'error': str(e),
                'parse_success': False
            }
            result['should_discard'] = None  # Unknown — API error
            result['discard_reason'] = f"Critique API call failed: {e}"
            result['is_feasible'] = None  # Unknown — never evaluated

        # Log to wandb
        if self.use_wandb and result.get('critique_response', {}).get('parse_success'):
            cr = result['critique_response']
            wandb.log({
                "scores/overall": cr['overall_score'],
                "scores/correctness": cr['scores'].get('correctness', 0),
                "scores/executability": cr['scores'].get('executability', 0),
                "scores/api_usage": cr['scores'].get('api_usage', 0),
                "scores/readability": cr['scores'].get('readability', 0),
                "scores/completeness": cr['scores'].get('completeness', 0),
                "feasibility/is_feasible": cr['feasibility']['is_feasible'],
                "feasibility/confidence": cr['feasibility']['confidence'],
                "decision/should_discard": cr['should_discard'],
            })

        return result

    def _process_and_record(self, record: dict, pbar, shard_prefix: str):
        """
        Process one record, then commit it to the checkpoint.

        A failed record is still marked processed so a restart doesn't retry it
        forever; the API-level retries already happened inside the client.
        """
        unique_id = record['_unique_id']

        cost = self.token_counter.get_cost()
        pbar.set_description(
            f"{shard_prefix}Processed: {len(self.processed_ids)} | "
            f"Cost: ${cost['total_cost']:.3f}"
        )

        try:
            record_clean = {k: v for k, v in record.items() if k != '_unique_id'}
            result = self.process_single_record(record_clean)

            if result:
                with self._state_lock:
                    self.results.append(result)
                    self.processed_ids.add(unique_id)

                overall_score = result.get('critique_response', {}).get('overall_score', 'N/A')
                is_feasible = result.get('is_feasible', 'N/A')
                should_discard = result.get('should_discard', 'N/A')
                logger.info(f"  ✅ {unique_id}: score={overall_score}, "
                            f"feasible={is_feasible}, discard={should_discard}")

        except Exception as e:
            logger.error(f"Error processing {unique_id}: {e}")
            with self._state_lock:
                self.processed_ids.add(unique_id)

        # Checkpoint after every record — an interrupted shard loses at most one.
        with self._state_lock:
            self._save_checkpoint()
        self._print_current_cost()
        pbar.update(1)

    def run(self):
        """Run the FIM completion and critique pipeline."""
        # Load input data directly (no preprocessing needed)
        logger.info(f"📂 Loading input data from {self.input_path}")
        with open(self.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_records_in_file = len(data)

        # Get shard range
        shard_start, shard_end = self._get_shard_range(total_records_in_file)
        data = data[shard_start:shard_end]

        logger.info(f"📊 Sharding info:")
        logger.info(f"   Total records in file: {total_records_in_file}")
        logger.info(f"   This shard's range: [{shard_start}, {shard_end})")
        logger.info(f"   Records in this shard: {len(data)}")

        # Generate unique IDs using the SAME module-level function
        # that process_single_record() uses — ensures checkpoint consistency.
        for record in data:
            record['_unique_id'] = make_unique_id(record)

        remaining_records = [r for r in data if r['_unique_id'] not in self.processed_ids]

        logger.info(f"   Already processed: {len(self.processed_ids)}")
        logger.info(f"   Remaining: {len(remaining_records)}")

        self._print_current_cost()

        # Build shard prefix for progress bar
        shard_prefix = ""
        if self.shard_id is not None:
            shard_prefix = f"[Shard {self.shard_id}/{self.total_shards}] "
        if self.concurrency > 1:
            logger.info(f"⚡ In-process concurrency: {self.concurrency} requests in flight")

        # Process records
        with tqdm(total=len(remaining_records), desc=f"{shard_prefix}Processing") as pbar:
            if self.concurrency == 1:
                for record in remaining_records:
                    self._process_and_record(record, pbar, shard_prefix)
            else:
                with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                    futures = [
                        pool.submit(self._process_and_record, record, pbar, shard_prefix)
                        for record in remaining_records
                    ]
                    for fut in as_completed(futures):
                        # _process_and_record swallows per-record errors; this only
                        # re-raises something catastrophic (e.g. KeyboardInterrupt).
                        fut.result()

        # Save final output
        logger.info(f"\n💾 Saving {len(self.results)} results to {self.output_path}")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        # Print summary
        self._print_summary()

        if self.use_wandb:
            finish_wandb()

    def _print_summary(self):
        """Print processing summary with feasibility and discard statistics."""
        print("\n" + "=" * 60)
        print("📊 Processing Summary")
        if self.shard_id is not None:
            print(f"   Shard: {self.shard_id}/{self.total_shards}")
        print("=" * 60)

        print(f"\n📈 Statistics:")
        print(f"   Total processed: {len(self.processed_ids)}")
        print(f"   Results saved: {len(self.results)}")

        self.token_counter.print_cost_summary()

        if self.results:
            # Feasibility stats (only count records where feasibility was evaluated)
            feasible_count = sum(1 for r in self.results if r.get('is_feasible') is True)
            infeasible_count = sum(1 for r in self.results if r.get('is_feasible') is False)
            unknown_feasibility = sum(1 for r in self.results if r.get('is_feasible') is None)
            evaluated = feasible_count + infeasible_count
            print(f"\n🔍 Feasibility:")
            if evaluated > 0:
                print(f"   Feasible: {feasible_count} ({feasible_count / evaluated * 100:.1f}%)")
                print(f"   Infeasible: {infeasible_count} ({infeasible_count / evaluated * 100:.1f}%)")
            if unknown_feasibility:
                print(f"   Unknown (API error): {unknown_feasibility}")

            # Discard stats
            discard_count = sum(1 for r in self.results if r.get('should_discard') is True)
            keep_count = sum(1 for r in self.results if r.get('should_discard') is False)
            unknown_count = sum(1 for r in self.results if r.get('should_discard') is None)
            print(f"\n🗑️ Discard Decision:")
            print(f"   Keep: {keep_count}")
            print(f"   Discard: {discard_count}")
            if unknown_count:
                print(f"   Unknown (API error): {unknown_count}")

            # Score distribution (only for successfully parsed critiques)
            scores = [r.get('critique_response', {}).get('overall_score', 0)
                      for r in self.results
                      if r.get('critique_response', {}).get('parse_success')]

            if scores:
                avg_score = sum(scores) / len(scores)
                print(f"\n📊 Score Statistics:")
                print(f"   Average overall score: {avg_score:.2f}")
                score_dist = {}
                for s in scores:
                    score_dist[s] = score_dist.get(s, 0) + 1
                print(f"   Score distribution:")
                for score in sorted(score_dist.keys()):
                    print(f"      Score {score}: {score_dist[score]} ({score_dist[score] / len(scores) * 100:.1f}%)")

            # Parse success rate
            fim_success = sum(1 for r in self.results if r.get('fim_response', {}).get('parse_success'))
            critique_success = sum(1 for r in self.results if r.get('critique_response', {}).get('parse_success'))
            n = len(self.results)
            print(f"\n📋 Parse Success Rate:")
            print(f"   FIM: {fim_success}/{n} ({fim_success / n * 100:.1f}%)")
            print(f"   Critique: {critique_success}/{n} ({critique_success / n * 100:.1f}%)")

        print("\n" + "=" * 60)




# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Single-function FIM completion and critique (step 4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Whole dataset, one process, paths from config.yaml:
  python single_function/step_4_fim_and_critique.py

  # One shard of 200 (launch the rest with scripts/run_step4_parallel.sh):
  python single_function/step_4_fim_and_critique.py --shard 1 --total-shards 200

  # Fewer processes, more requests in flight per process:
  python single_function/step_4_fim_and_critique.py --shard 1 --total-shards 8 --concurrency 16

  # An explicit slice, e.g. to smoke-test 20 records before spending real money:
  python single_function/step_4_fim_and_critique.py --start-idx 0 --end-idx 20 --print-response
        '''
    )

    add_config_arg(parser)

    # Input/Output — default to the paths derived from config.yaml
    parser.add_argument("--input", "-i", default=None,
                        help="Override step 3's *_functions.json")
    parser.add_argument("--output", "-o", default=None,
                        help="Override <work_dir>/single_function/step_4_fim_critique.json")
    parser.add_argument("--checkpoint", "-c", default=None,
                        help="Checkpoint path (default: <output stem>_checkpoint.json)")

    # Model configuration
    parser.add_argument("--model", "-m", default=None, help="Override llm.model")
    parser.add_argument("--fim-temperature", type=float, default=None)
    parser.add_argument("--critique-temperature", type=float, default=None)

    # Processing options
    parser.add_argument("--wait", "-w", type=float, default=None,
                        help="Seconds to sleep after each API call (override llm.wait_seconds)")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Requests in flight within this process (override sharding.concurrency)")
    parser.add_argument("--print-response", "-p", action="store_true",
                        help="Print every API response — use on a small slice only")

    # Sharding
    parser.add_argument("--shard", type=int, default=None, help="Shard ID (1-indexed)")
    parser.add_argument("--total-shards", type=int, default=None, help="Total number of shards")
    parser.add_argument("--start-idx", type=int, default=None, help="Start index (inclusive)")
    parser.add_argument("--end-idx", type=int, default=None, help="End index (exclusive)")

    # W&B
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)

    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = derive_paths(cfg)
    llm = cfg["llm"]

    input_path = Path(args.input) if args.input else paths["single_step3_functions"]
    output_path = Path(args.output) if args.output else paths["single_step4_out"]

    if not input_path.exists():
        raise SystemExit(
            f"Error: input not found: {input_path}\n"
            "Run single_function/step_3_select_functions.py first."
        )

    # Checkpoint defaults to <output stem>_checkpoint.json; the shard suffix is
    # appended to both inside the pipeline, keeping shards from colliding.
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        output_path.parent / f"{output_path.stem}_checkpoint.json"
    )

    # Validate sharding arguments
    if args.shard is not None and args.total_shards is None:
        parser.error("--shard requires --total-shards")
    if args.total_shards is not None and args.shard is None:
        parser.error("--total-shards requires --shard")
    if args.shard is not None and (args.start_idx is not None or args.end_idx is not None):
        parser.error("Cannot use --shard with --start-idx or --end-idx")

    use_wandb = args.wandb or cfg["wandb"]["enabled"]
    if use_wandb and not WANDB_AVAILABLE:
        logger.warning("⚠️ W&B requested but not installed — continuing without it")
        use_wandb = False

    pipeline = FIMCompletionPipeline(
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
