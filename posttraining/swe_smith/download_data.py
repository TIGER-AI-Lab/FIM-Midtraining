#!/usr/bin/env python3
"""
Build the SWE-Smith training file: resolved agent trajectories in chat format.

Three filters, in order:
  1. the `xml` split — the trajectory format SWE-Smith trains on
  2. resolved == True — only trajectories whose patch actually fixed the issue.
     An unresolved trajectory is a recording of a failure; training on it
     teaches the model to fail confidently.
  3. Claude-3.7 trajectories if there are enough of them, else all resolved ones

Then a seeded downsample to --num-trajs, so the run is reproducible.

    python download_data.py                     # 8000 trajectories (what we ran)
    python download_data.py --num-trajs 0       # keep every resolved trajectory
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

DATASET_ID = "SWE-bench/SWE-smith-trajectories"
SPLIT = "xml"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DATASET_ID, help="HF dataset id")
    parser.add_argument("--split", default=SPLIT, help="Trajectory format split")
    parser.add_argument("--num-trajs", type=int, default=8000,
                        help="Downsample to this many (0 = keep all resolved)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./data/swe_smith_trajectories.jsonl")
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    print(f"Total trajectories:    {len(ds):,}")

    resolved = [row for row in ds if row.get("resolved", False)]
    print(f"Resolved:              {len(resolved):,}")

    claude37 = [r for r in resolved if "claude-3-7" in (r.get("model") or "").lower()]
    print(f"Claude-3.7 resolved:   {len(claude37):,}")

    # Prefer the stronger teacher, but only if it alone can fill the budget —
    # otherwise mixing in the rest beats training on too little data.
    if args.num_trajs and len(claude37) >= args.num_trajs:
        selected = claude37
        print("Using Claude-3.7 trajectories only")
    else:
        selected = resolved
        print("Using all resolved trajectories")

    if args.num_trajs and len(selected) > args.num_trajs:
        random.seed(args.seed)
        selected = random.sample(selected, args.num_trajs)
        print(f"Downsampled to:        {len(selected):,} (seed {args.seed})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in selected:
            messages = row["messages"]
            # The Hub stores `messages` as a JSON *string*; torchtune's
            # chat_dataset expects a real list. Skipping this parse yields a
            # dataset that loads fine and trains on nonsense.
            if isinstance(messages, str):
                messages = json.loads(messages)
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(selected):,} trajectories -> {out_path}")


if __name__ == "__main__":
    main()
