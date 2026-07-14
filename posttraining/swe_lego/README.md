# SWE-Lego Post-Training

Used for the Qwen3-8B rows of Table 1. Reproduces the cross-base-model result.

## Why this pipeline for Qwen3-8B

Not a free choice. SWE-Lego's trajectories exceed the 32K context window the R2E-Gym and SWE-Smith scaffolds are built around, so Qwen3-8B (40,960-token context) needs a scaffold that can carry them. This means the Qwen3-8B comparison **varies the post-training pipeline at the same time as the base model.** Read the result as "the gain is not specific to the Qwen2.5-Coder + R2E-Gym/SWE-Smith pairing," not as a clean cross-family guarantee. The paper says as much in Section 4.2.

## Setup

```bash
git clone https://github.com/hiyouga/LLaMA-Factory
cd LLaMA-Factory
pip install -e ".[torch,metrics,deepspeed,liger-kernel]"
```

## Data

Two splits, both filtered to `resolved` (trajectories whose patch actually fixed the issue):

- `SWE-Lego/SWE-Lego-Synthetic-Data`
- the SWE-Lego real-data trajectories

See [`download_data.py`](download_data.py), then register both in `data/dataset_info.json`.

`turn_mask: true` in the config masks the loss on failed turns, so only successful agent steps contribute gradient.

## Run

```bash
cp swe_lego_posttrain.yaml <LLaMA-Factory>/
cd <LLaMA-Factory>

# baseline
llamafactory-cli train swe_lego_posttrain.yaml \
  model_name_or_path=Qwen/Qwen3-8B output_dir=saves/swe-lego-baseline

# ours
llamafactory-cli train swe_lego_posttrain.yaml \
  model_name_or_path=saves/fim-midtrain/checkpoint-XXX output_dir=saves/swe-lego-fim
```

## The one deviation from upstream

**2 epochs, not the official 4.** Four epochs overfits the FIM-midtrained Qwen3-8B base in our setup. Every other hyperparameter is the official SWE-Lego recipe. This is applied to *both* arms (baseline and ours), so the comparison stays fair.

## Reproduces

| Config | Paper row (Table 1) |
|---|---|
| Qwen3-8B + SWE-Lego | `+ SWE-Lego (reproduced)` — 31.80 Verified / 27.30 Lite |
| Qwen3-8B mid-trained + SWE-Lego | `+ FIM-Midtrain + SWE-Lego` — 35.00 / 32.70 |
