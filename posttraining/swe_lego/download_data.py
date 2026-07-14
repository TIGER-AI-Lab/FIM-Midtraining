#!/usr/bin/env python3
"""
Download the SWE-Lego trajectories to local JSONL.

Keeps only the `resolved` split — trajectories whose patch actually fixed the
issue. Unresolved trajectories are not training signal.

    python download_data.py --output-dir ./data
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

# The synthetic half. The real-data half is the other dataset listed in
# swe_lego_posttrain.yaml; pass it with --dataset to fetch that one too.
DATASET_ID = "SWE-Lego/SWE-Lego-Synthetic-Data"
SPLIT = "resolved"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_ID, help="HF dataset id")
    parser.add_argument("--split", default=SPLIT, help="Split to keep (default: resolved)")
    parser.add_argument("--output-dir", default="./data", help="Where to write the JSONL")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split)
    out_path = out_dir / f"{args.dataset.split('/')[-1]}_{args.split}.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            # Keep only what the trainer reads.
            f.write(json.dumps(
                {"instance_id": row.get("instance_id"), "messages": row.get("messages")},
                ensure_ascii=False,
            ) + "\n")

    print(f"{len(ds):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
