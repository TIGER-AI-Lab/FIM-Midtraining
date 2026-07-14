# SWE-Smith Post-Training

Reproduces the SWE-Smith rows of Table 1 (Qwen2.5-Coder-7B-Instruct). This pipeline exists in the paper to show the mid-training gain is **not tuned to one post-training data distribution**: swapping R2E-Gym for SWE-Smith on the same base still gains (+5.3 on Verified — larger than under R2E-Gym — though only +0.5 on Lite).

This is the one pipeline that does **not** use LLaMA-Factory. SWE-Smith's official recipe is [torchtune](https://github.com/pytorch/torchtune), and we follow it unmodified.

## Setup

```bash
pip install torchtune torchao
```

The base checkpoint directory must contain the tokenizer (`vocab.json`, `merges.txt`) alongside the safetensors shards — torchtune loads both from the same place. A stock `huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct` gives you this; so does a LLaMA-Factory mid-training output.

## Data

```bash
python download_data.py
```

Pulls `SWE-bench/SWE-smith-trajectories` (`xml` split), keeps only **resolved** trajectories (a trajectory whose patch did not fix the issue is a recording of a failure — training on it teaches the model to fail confidently), prefers Claude-3.7 trajectories when there are enough, and downsamples to **8,000** with seed 42. That count is what we ran; `--num-trajs 0` keeps every resolved trajectory instead.

## Run

Twice, identical except for the starting checkpoint:

```bash
# baseline
MODEL_DIR=/path/to/Qwen2.5-Coder-7B-Instruct \
EXP_NAME=swe-smith-baseline ./run_posttrain.sh

# ours
MODEL_DIR=/path/to/fim-midtrain/checkpoint-XXX \
EXP_NAME=swe-smith-fim ./run_posttrain.sh
```

Everything else is an env var (`NUM_GPUS`, `GRAD_ACCUM`, `LR`, `EPOCHS`, `MAX_SEQ_LEN`) — you should not need to edit the YAML to switch arms or resize.

## Hyperparameters

Paper Appendix C, Table 8 — the upstream SWE-Smith recipe. Note how far these sit from the other two pipelines; that is the point of running it.

| | |
|---|---|
| Framework | torchtune (`full_finetune_distributed`) |
| Learning rate | **1e-4** — 10x the R2E-Gym rate |
| Warmup | 5 steps |
| Weight decay | 0.01 |
| Epochs | 3 |
| Per-device batch size | 1 |
| Gradient accumulation | 4 |
| **Effective batch size** | **32** (assumes 8 GPUs) |
| Sequence length | 32,768 |
| Optimizer | AdamW (fused), bf16 |

If you change `NUM_GPUS`, change `GRAD_ACCUM` with it to keep the effective batch size at 32.

## Reproduces

| Config | Paper row (Table 1) |
|---|---|
| 7B stock + SWE-Smith | `+ SWE-Smith (reproduced)` — 12.30 Verified / 14.20 Lite |
| 7B mid-trained + SWE-Smith | `+ FIM-Midtrain + SWE-Smith` — 17.60 / 14.70 |

## Evaluate

[`../../evaluation/swebench/`](../../evaluation/swebench).
