# Terminal-Bench

> **Status: not yet released.** This directory is a placeholder — the evaluation
> for this benchmark was run outside this repository and the scripts have not
> been recovered. See "What belongs here" below.

## What this benchmark is for

Agentic tasks in a real terminal sandbox. Needs Docker.

**Role in the paper:** Agent OOD — an agent benchmark that is not SWE-Bench. Note the Instruct model scores 0.00 here (it cannot drive an agent loop at all), so both trained rows are improvements over a floor, not a recovery toward a ceiling.

## Numbers to reproduce

Paper Table 2 (capability preservation), Qwen2.5-Coder-14B-Instruct + R2E-Gym.
All three rows use the same checkpoints; only the benchmark differs.

| Setting | Score |
|---|---|
| Instruct (ceiling) | 0.00 |
| + R2E-Gym | 2.41 |
| + FIM Mid-Train + R2E-Gym | **3.66** |

## What belongs here

A runner that takes a **post-trained checkpoint** and emits a single score, for
each of the three arms above. In practice that means:

1. **Serve the checkpoint.** Reuse
   [`../swebench/start_vllm_server.sh`](../swebench/start_vllm_server.sh) — every
   benchmark here talks to an OpenAI-compatible endpoint, so the serving step is
   identical and should not be re-implemented per benchmark.
2. **Drive the upstream harness** against that endpoint: https://github.com/laude-institute/terminal-bench
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

https://github.com/laude-institute/terminal-bench
