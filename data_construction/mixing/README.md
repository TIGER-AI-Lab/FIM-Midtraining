# Corpus Mixing

> **The mixed corpus itself is released** — [`all_merged_400k.jsonl` on the Hub](https://huggingface.co/datasets/TIGER-Lab/FIM-Midtraining-400K) *is* the 80/15/5 mixture used for the main results (320K single + 60K pairs + 20K triples, shuffled), so you only need this step if you are rebuilding a corpus of your own or reproducing the mixture ablations. A standalone mixing script has not been recovered; everything needed to write one is described below.

## What belongs here

The main results train on a **mixture**, not on any single split. The pipeline in `../single_function` and `../multi_function` emits three separate JSONL files; this step samples them into one corpus.

The recipe from Table 4 block (C):

| Split | Share | Source file |
|---|---|---|
| single-function | **80%** | `<work_dir>/single_function/sft/single_function_fim_sft.jsonl` |
| pairs (k=2) | **15%** | `<work_dir>/multi_function/sft/multi_function_fim_sft_pairs.jsonl` |
| triples (k=3) | **5%** | `<work_dir>/multi_function/sft/multi_function_fim_sft_triples.jsonl` |

Output: one shuffled JSONL, registered in LLaMA-Factory as `fim_midtrain` (see [`../../midtraining/dataset_info.json`](../../midtraining/dataset_info.json)).

The released corpus is ~400K samples / ~2.6B tokens under the Qwen2.5-Coder tokenizer: ~320K single (~2.0B tokens), ~60K pairs (~0.4B), ~20K triples (~0.2B).

## Why the ratios are what they are

The mix is an empirical result, not a guess (Table 4C, 7B + R2E-Gym):

| Mixture | SWE-Bench-Verified | SWE-Bench-Lite | Avg |
|---|---|---|---|
| single only | 17.00 | 14.20 | 15.60 |
| 85% single + 15% pair | 17.20 | 14.60 | 15.90 |
| 95% single + 5% triple | 17.00 | 14.40 | 15.70 |
| **80/15/5 (ours)** | **17.40** | **14.80** | **16.10** |

Pairs help; triples on their own are essentially neutral; the three-way mix is best. Marginal return falls off as group size grows — coupling gets harder to maintain when more of the file is masked at once.

So whatever is written here must be able to reproduce all four rows, i.e. the ratios need to be a parameter, not a constant.

## Interface

Something like:

```bash
python mix.py \
  --single  <work_dir>/single_function/sft/single_function_fim_sft.jsonl \
  --pairs   <work_dir>/multi_function/sft/multi_function_fim_sft_pairs.jsonl \
  --triples <work_dir>/multi_function/sft/multi_function_fim_sft_triples.jsonl \
  --ratios 0.80,0.15,0.05 \
  --seed 42 \
  --output <work_dir>/fim_midtrain_mixed.jsonl
```

Worth getting right: with a fixed total budget the ratios are a *sampling* problem, and whether you take the highest-`fim_score` items or sample uniformly changes the corpus. The paper's ablations hold the budget fixed precisely so that rows differ in *which* functions are selected, not *how many*.
