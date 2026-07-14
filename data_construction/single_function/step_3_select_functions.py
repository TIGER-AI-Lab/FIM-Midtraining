#!/usr/bin/env python3
"""
Step 3 (single-function) — pick which individual functions to mask.

Builds a dependency graph per file and keeps the functions whose FIM score
clears `selection.score_threshold`: substantial enough to be worth predicting,
but recoverable from the surrounding file. See common/dep_graph.py for the
scoring itself.

Writes two files:
  <out>.json            one record per source file, with a `mask_targets` list
  <out>_functions.json  one record per selected function, each carrying the
                        whole file with that one body replaced by
                        `# <MASKED_FUNCTION_BODY>` — this is what step 4 eats.

    python single_function/step_3_select_functions.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import (  # noqa: E402
    add_config_arg, derive_paths, load_config, selection_config,
)
from common.dep_graph import (  # noqa: E402
    postprocess_results, print_distribution_stats, process_samples, save_results,
)


def main():
    parser = argparse.ArgumentParser(
        description="Select single-function FIM mask targets via dependency-graph scoring."
    )
    add_config_arg(parser)
    parser.add_argument("--input", "-i", default=None,
                        help="Override <work_dir>/extracted_python_files.json (step 2 output)")
    parser.add_argument("--output", "-o", default=None,
                        help="Override <work_dir>/single_function/step_3_selected.json")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress the per-file breakdown")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = derive_paths(cfg)

    input_path = Path(args.input) if args.input else paths["extracted_files"]
    output_path = Path(args.output) if args.output else paths["single_step3_out"]

    if not input_path.exists():
        sys.exit(f"Error: input not found: {input_path}\nRun common/step_2_extract_python_files.py first.")

    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples from {input_path}")

    selection = selection_config(cfg)
    if selection:
        print(f"Selection config: {selection}")

    # Stage 1 — score every file, attach its mask targets.
    results = process_samples(samples, config=selection, verbose=not args.quiet)
    save_results(results, str(output_path))

    # Stage 2 — explode into one entry per selected function, each with the
    # masked version of its file. This is the file step 4 consumes.
    func_entries = postprocess_results(results)
    func_output_path = output_path.parent / f"{output_path.stem}_functions{output_path.suffix}"
    save_results(func_entries, str(func_output_path))

    # Stage 3 — distribution of loc / complexity / inferability / fim_score.
    print_distribution_stats(func_entries)

    print(f"\n✅ Step 3 done. Feed step 4 with:\n   {func_output_path}")


if __name__ == "__main__":
    main()
