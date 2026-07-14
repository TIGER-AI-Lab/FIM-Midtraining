# R2E-Gym Post-Training

The primary post-training pipeline, used for both Qwen2.5-Coder sizes. Reproduces the R2E-Gym rows of Table 1.

## Setup

```bash
git clone https://github.com/hiyouga/LLaMA-Factory
cd LLaMA-Factory
pip install -e ".[torch,metrics,deepspeed,liger-kernel]"
```

## Data

`R2E-Gym/R2EGym-SFT-Trajectories` from the Hub. LLaMA-Factory can load it by id once registered in `data/dataset_info.json`; use [`download_data.py`](download_data.py) if the training node has no internet.

## Run

Two runs, identical except for the starting checkpoint:

```bash
cp r2egym_posttrain.yaml <LLaMA-Factory>/
cd <LLaMA-Factory>

# baseline: stock instruct model
llamafactory-cli train r2egym_posttrain.yaml \
  model_name_or_path=Qwen/Qwen2.5-Coder-7B-Instruct \
  output_dir=saves/r2egym-baseline-7b

# ours: FIM mid-trained checkpoint
llamafactory-cli train r2egym_posttrain.yaml \
  model_name_or_path=saves/fim-midtrain/checkpoint-XXX \
  output_dir=saves/r2egym-fim-7b
```

Swap in `Qwen2.5-Coder-14B-Instruct` (and its mid-trained checkpoint) for the 14B rows. Nothing else changes.

## Configs

| Config | What it is |
|---|---|
| [`r2egym_posttrain.yaml`](r2egym_posttrain.yaml) | **The reference recipe** — edit `model_name_or_path`, run |
| [`FIM_Posttrain_7B.yaml`](FIM_Posttrain_7B.yaml) | As run for [TIGER-Lab/FIM-7B](https://huggingface.co/TIGER-Lab/FIM-7B), starting from [FIM-Mid-7B](https://huggingface.co/TIGER-Lab/FIM-Mid-7B) |
| [`FIM_Posttrain_14B.yaml`](FIM_Posttrain_14B.yaml) | As run for [TIGER-Lab/FIM-14B](https://huggingface.co/TIGER-Lab/FIM-14B), starting from [FIM-Mid-14B](https://huggingface.co/TIGER-Lab/FIM-Mid-14B) |

The as-run configs start from the released mid-trained checkpoints on the Hub, so they reproduce the paper's `+ FIM-Midtrain + R2E-Gym` rows without running mid-training yourself.

## Reproduces

| Config | Paper row (Table 1) |
|---|---|
| 7B stock + R2E-Gym | `+ R2E-Gym (reproduced)` — 15.00 Verified / 11.33 Lite |
| 7B mid-trained + R2E-Gym | `+ FIM-Midtrain + R2E-Gym` — 17.80 / 15.00 |
| 14B stock + R2E-Gym | `+ R2E-Gym (reproduced)` — 26.20 / 18.00 |
| 14B mid-trained + R2E-Gym | `+ FIM-Midtrain + R2E-Gym` — 29.20 / 22.00 |

Our reproduced baseline is below the officially reported R2E-Gym number at 7B (15.00 vs 19.00 on Verified). The paper reports both, and computes the delta against **our own reproduction**, since that is the only apples-to-apples comparison — the mid-trained run uses this same config.

## Evaluate

[`../../evaluation/swebench/`](../../evaluation/swebench).
