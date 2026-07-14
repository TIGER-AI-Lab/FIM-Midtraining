#!/usr/bin/env python3
"""
Download the R2E-Gym SFT trajectories to a local JSONL.

Only needed if you want a local copy — LLaMA-Factory can also pull the dataset
straight from the Hub by id. Useful when the training node has no internet.

    python download_data.py --output-dir ./data
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

DATASET_ID = "R2E-Gym/R2EGym-SFT-Trajectories"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_ID, help="HF dataset id")
    parser.add_argument("--output-dir", default="./data", help="Where to write the JSONL")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset)
    for split, rows in ds.items():
        out_path = out_dir / f"{args.dataset.split('/')[-1]}_{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
