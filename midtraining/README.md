# FIM Mid-Training

Takes a base model and the FIM corpus from [`../data_construction`](../data_construction), and produces the mid-trained checkpoint that every post-training run starts from.

This is **one reference script, not a script per experiment.** The paper applies the same recipe to all three base models — swap `model_name_or_path` (and `template` for Qwen3) and nothing else changes.

## Setup

Mid-training uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory):

```bash
git clone https://github.com/hiyouga/LLaMA-Factory
cd LLaMA-Factory
pip install -e ".[torch,metrics,deepspeed,liger-kernel]"
pip install flash-attn --no-build-isolation
```

## Register the corpus

Copy your FIM JSONL into LLaMA-Factory's `data/` directory, then paste the entries from [`dataset_info.json`](dataset_info.json) into LLaMA-Factory's own `data/dataset_info.json`.

The main results train on the **80% single / 15% pair / 5% triple** mixture (paper Table 4, block C). `data_construction` emits the three splits separately; see [`../data_construction/mixing/`](../data_construction/mixing) for the mixing step.

## Run

```bash
cp configs/fim_midtrain.yaml <LLaMA-Factory>/
cd <LLaMA-Factory>
llamafactory-cli train fim_midtrain.yaml
```

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
