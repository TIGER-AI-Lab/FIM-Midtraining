# FIM Mid-Training

Takes a base model and the FIM corpus — the released [TIGER-Lab/FIM-Midtraining-400K](https://huggingface.co/datasets/TIGER-Lab/FIM-Midtraining-400K), or your own build from [`../data_construction`](../data_construction) — and produces the mid-trained checkpoint that every post-training run starts from.

The recipe is **one reference config, not a config per experiment.** The paper applies it to all three base models — swap `model_name_or_path` and nothing else changes. The exact per-model configs behind the released `FIM-Mid-*` checkpoints are also included (see [Configs](#configs)).

The three released mid-trained checkpoints, if you want to skip this stage entirely: [FIM-Mid-7B](https://huggingface.co/TIGER-Lab/FIM-Mid-7B) · [FIM-Mid-8B](https://huggingface.co/TIGER-Lab/FIM-Mid-8B) · [FIM-Mid-14B](https://huggingface.co/TIGER-Lab/FIM-Mid-14B)

## Setup

Mid-training uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory):

```bash
git clone https://github.com/hiyouga/LLaMA-Factory
cd LLaMA-Factory
pip install -e ".[torch,metrics,deepspeed,liger-kernel]"
pip install flash-attn --no-build-isolation
```

## Get and register the corpus

Download the released corpus into LLaMA-Factory's `data/` directory, then paste the entries from [`dataset_info.json`](dataset_info.json) into LLaMA-Factory's own `data/dataset_info.json`:

```bash
huggingface-cli download TIGER-Lab/FIM-Midtraining-400K all_merged_400k.jsonl \
  --repo-type dataset --local-dir <LLaMA-Factory>/data/
```

The main results train on the **80% single / 15% pair / 5% triple** mixture (paper Table 4, block C) — `all_merged_400k.jsonl` *is* that mixture (320K single + 60K pairs + 20K triples, shuffled). The three unmixed splits are released alongside it for ablations, and [`../data_construction`](../data_construction) rebuilds everything from the repo list if you want a corpus of your own.

## Run

```bash
cp configs/fim_midtrain.yaml <LLaMA-Factory>/
cd <LLaMA-Factory>
llamafactory-cli train fim_midtrain.yaml
```

## Configs

| Config | What it is |
|---|---|
| [`configs/fim_midtrain.yaml`](configs/fim_midtrain.yaml) | **The reference recipe** — edit `model_name_or_path`, run |
| [`configs/FIM_Midtrain_7B.yaml`](configs/FIM_Midtrain_7B.yaml) | As run for [TIGER-Lab/FIM-Mid-7B](https://huggingface.co/TIGER-Lab/FIM-Mid-7B) |
| [`configs/FIM_Midtrain_8B.yaml`](configs/FIM_Midtrain_8B.yaml) | As run for [TIGER-Lab/FIM-Mid-8B](https://huggingface.co/TIGER-Lab/FIM-Mid-8B) |
| [`configs/FIM_Midtrain_14B.yaml`](configs/FIM_Midtrain_14B.yaml) | As run for [TIGER-Lab/FIM-Mid-14B](https://huggingface.co/TIGER-Lab/FIM-Mid-14B) |

The as-run configs preserve the exact hyperparameters of the released checkpoints (they differ from the reference only in bookkeeping fields like `save_steps`, plus `rope_scaling: yarn`); their dataset references point at the released corpus.

## What to change per base model

| Base model | `model_name_or_path` | `template` |
|---|---|---|
| Qwen2.5-Coder-7B-Instruct | `Qwen/Qwen2.5-Coder-7B-Instruct` | `qwen` |
| Qwen2.5-Coder-14B-Instruct | `Qwen/Qwen2.5-Coder-14B-Instruct` | `qwen` |
| Qwen3-8B (base, **not** Instruct) | `Qwen/Qwen3-8B` | `qwen3` |

Everything else is held fixed across models — that is the point of the experiment.

## Hyperparameters

Paper Appendix C, Table 7. Reproduced here so you can sanity-check a run without opening the paper:

| | |
|---|---|
| Optimizer | AdamW, bf16 |
| Learning rate | 1e-5, cosine, warmup ratio 0.1 |
| Weight decay | 0.05 |
| Epochs | 1 |
| Per-device batch size | 1 |
| Gradient accumulation | 16 |
| **Effective batch size** | **128** (assumes 8 GPUs) |
| Sequence length | 32,768 |

If you run on a different GPU count, adjust `gradient_accumulation_steps` to keep the effective batch size at 128.

## Compute

8x H100 80GB, single node. At 32K context the three memory options in the config (`flash_attn: fa2`, `enable_liger_kernel`, `use_unsloth_gc`) are not optional — dropping any one of them OOMs in our setup.

## Next

The resulting checkpoint is the input to [`../posttraining`](../posttraining). The paper does **not** evaluate mid-training-only checkpoints: a FIM-only model has degraded instruction-following and cannot be fairly compared against an instruction-tuned baseline. Every reported gain is one that *survives* post-training.
