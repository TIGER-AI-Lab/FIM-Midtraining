# Agentic Post-Training

The paper's claim is not "FIM mid-training makes a good model" — it is that **the gain survives an off-the-shelf agentic post-training pipeline that we do not modify at all.** So every pipeline here is the upstream recipe, run twice:

| | starting checkpoint |
|---|---|
| **baseline** | the stock instruct model |
| **ours** | the FIM mid-trained checkpoint from [`../midtraining`](../midtraining) |

Everything else is held identical. The difference between the two runs is the number the paper reports.

Each directory holds **one reference config**, not one per model size. Point it at a different base model and it reproduces the corresponding row.

## The three pipelines

| Pipeline | Base model in the paper | Framework | Paper table |
|---|---|---|---|
| [`r2egym/`](r2egym) | Qwen2.5-Coder-7B / 14B-Instruct | LLaMA-Factory | Table 1, all Qwen2.5-Coder rows |
| [`swe_smith/`](swe_smith) | Qwen2.5-Coder-7B-Instruct | **torchtune** | Table 1, SWE-Smith rows |
| [`swe_lego/`](swe_lego) | Qwen3-8B | LLaMA-Factory | Table 1, Qwen3-8B rows |

Why three: R2E-Gym is the primary pipeline. SWE-Smith exists to show the gain is not tuned to one post-training data distribution. SWE-Lego is used for Qwen3-8B because its trajectories exceed the 32K context the other two scaffolds are built around — which means the Qwen3-8B comparison varies the pipeline *and* the base model together. That confound is stated in the paper (Section 4.2) and should not be read as a clean cross-family result.

## Hyperparameters

Paper Appendix C, Table 8. R2E-Gym and SWE-Smith follow their official released scripts; SWE-Lego follows the official recipe **except for the epoch count**.

| | R2E-Gym | SWE-Smith | SWE-Lego |
|---|---|---|---|
| Framework | LLaMA-Factory | torchtune | LLaMA-Factory |
| Learning rate | 1e-5 | 1e-4 | 1e-4 |
| Warmup | ratio 0.05 | 5 steps | ratio 0.1 |
| Weight decay | 0.0 | 0.01 | 0.01 |
| Epochs | 2 | 3 | **2** (official: 4) |
| Gradient accumulation | 1 | 4 | 8 |
| Effective batch size | 8 | 32 | 64 |
| Sequence length | 32,768 | 32,768 | 40,960 |

All on 8x H100 80GB, AdamW, bf16, cosine schedule.

**The one deliberate deviation:** SWE-Lego is trained for 2 epochs instead of the official 4. Four epochs overfits the FIM-midtrained Qwen3-8B base in our setup. Everything else is upstream.

## Output

The final checkpoint of each run is what [`../evaluation`](../evaluation) scores. The paper reports the mean over three independent evaluation seeds on that final checkpoint.
