# tau-bench

> **Status: not yet released.** This directory is a placeholder — the evaluation
> for this benchmark was run outside this repository and the scripts have not
> been recovered. See "What belongs here" below.

## What this benchmark is for

Tool-agent conversations in retail/airline domains, scored on task completion against a user simulator. Needs an API key for the user-simulator model.

**Role in the paper:** Tool use, fully outside the coding domain. This is one of the paper's two load-bearing transfer results: the mid-training corpus contains NO tool-use trajectories and NO non-Python data, so a gain here can only come from a structural prior installed at mid-training that survives post-training.

## Numbers to reproduce

Paper Table 2 (capability preservation), Qwen2.5-Coder-14B-Instruct + R2E-Gym.
All three rows use the same checkpoints; only the benchmark differs.

| Setting | Score |
|---|---|
| Instruct (ceiling) | 5.70 |
| + R2E-Gym | 3.40 |
| + FIM Mid-Train + R2E-Gym | **7.30** |

## What belongs here

A runner that takes a **post-trained checkpoint** and emits a single score, for
each of the three arms above. In practice that means:

1. **Serve the checkpoint.** Reuse
   [`../swebench/start_vllm_server.sh`](../swebench/start_vllm_server.sh) — every
   benchmark here talks to an OpenAI-compatible endpoint, so the serving step is
   identical and should not be re-implemented per benchmark.
2. **Drive the upstream harness** against that endpoint: https://github.com/sierra-research/tau-bench
   Do not reimplement the benchmark; the published numbers only mean something
   if they come from the official harness.
3. **Parse the harness output into one number** and write it somewhere
   comparable across the three arms.

The three arms to run:

| Arm | Checkpoint |
|---|---|
| Instruct (ceiling) | `Qwen/Qwen2.5-Coder-14B-Instruct` |
| post-training only | `../../posttraining/r2egym` output, started from the stock model |
| ours | `../../posttraining/r2egym` output, started from the mid-trained model |

## Upstream

https://github.com/sierra-research/tau-bench
