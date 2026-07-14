# SWE-Bench Evaluation (Verified + Lite)

The primary in-domain benchmark — Table 1 of the paper. This is the one evaluation with complete, working code in this repo.

## How it works

Three stages, three shells:

```
start_vllm_server.sh   serve the checkpoint over an OpenAI-compatible API
        |
run_rollout.sh         R2E-Gym's agent scaffold solves each instance in Docker,
        |              talking to the vLLM endpoint, and emits patches
score.sh               convert to SWE-Bench submission format, then score with
                       the OFFICIAL SWE-Bench harness (re-runs the real tests)
```

The number that goes in the paper comes from stage 3 — the official harness re-running each repository's own test suite against the agent's patch. R2E-Gym's internal reward is *not* what we report.

## Prerequisites

- **Docker**, running, with disk for the SWE-Bench images (hundreds of GB for Verified).
- **[R2E-Gym](https://github.com/R2E-Gym/R2E-Gym)** — upstream, unmodified. Provides the agent scaffold.
- **[SWE-bench](https://github.com/princeton-nlp/SWE-bench)** — upstream. Provides the scorer.
- **vLLM**, and a GPU to serve the checkpoint.

```bash
git clone https://github.com/R2E-Gym/R2E-Gym    && export R2E_GYM=$PWD/R2E-Gym
git clone https://github.com/princeton-nlp/SWE-bench && export SWE_BENCH=$PWD/SWE-bench
```

Follow each repo's own install instructions. We pass `--backend docker` and set `LLM_BASE_URL` explicitly, so **no patch to R2E-Gym is needed** — the scripts here work against a clean checkout.

## Run

```bash
# shell 1 — serve the post-trained checkpoint
./start_vllm_server.sh /path/to/r2egym-fim-7b/checkpoint-XXX

# shell 2 — roll out, then score
export R2E_GYM=... SWE_BENCH=... EXP_NAME=fim-7b

./run_rollout.sh verified
./score.sh verified ./traj_fim-7b_verified/<trajectory>.jsonl

./run_rollout.sh lite
./score.sh lite ./traj_fim-7b_lite/<trajectory>.jsonl
```

Then repeat the whole thing for the **baseline** checkpoint. The paper's reported gain is the difference between the two, and both arms must go through this identical harness — comparing your number against a *published* baseline instead of your own reproduction is how you end up reporting a gain that is really a harness difference.

## Protocol

The paper reports the **mean over three independent evaluation seeds** on the final checkpoint of each pipeline. Rollout temperature is 1.0, so re-running `run_rollout.sh` gives you a different seed; average three.

Verified is k=500 instances, Lite is k=300.

## Cost

This is slow. A single Verified pass is hundreds of Docker containers each running an agent loop for up to 100 steps, then hundreds more to score. Budget hours per pass, and remember the paper needs *three passes x two arms x two splits* per model.

## Knobs

Everything is an env var: `PORT`, `GPUS`, `MAX_LEN`, `TP_SIZE`, `MAX_WORKERS`, `MAX_STEPS_ABSOLUTE`, `TEMPERATURE`, `RUN_ID`. Defaults match what we ran.

One thing worth not changing: `MAX_LEN=65536`. Agent trajectories accumulate tool output (file contents, test logs) turn after turn; below ~32K the long rollouts get truncated mid-episode and silently score as failures, which reads as a worse model rather than a truncated one.
