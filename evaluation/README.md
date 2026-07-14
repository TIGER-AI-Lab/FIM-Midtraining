# Evaluation

Eight benchmarks in three groups. Every one of them scores a **post-trained** checkpoint — the paper deliberately never evaluates a mid-training-only model, because a FIM-only model has degraded instruction-following and could not be compared fairly against an instruction-tuned baseline. Every reported gain is one that survived post-training.

| Group | Benchmark | Paper table | Status |
|---|---|---|---|
| **Coding agent** (in-domain) | [SWE-Bench-Verified](swebench) | Table 1 | ✅ scripts here |
| | [SWE-Bench-Lite](swebench) | Table 1 | ✅ scripts here |
| **Non-agent coding** (regression check) | [LiveCodeBench](livecodebench) | Table 2 | ⬜ placeholder |
| | [OJBench](ojbench) | Table 2 | ⬜ placeholder |
| | [FullStackBench-EN](fullstackbench) | Table 2 | ⬜ placeholder |
| **Agent OOD** | [Terminal-Bench](terminal_bench) | Table 2 | ⬜ placeholder |
| **Tool use** (outside coding) | [τ-bench](tau_bench) | Table 2 | ⬜ placeholder |
| | [BFCL](bfcl) | Table 2 | ⬜ placeholder |

**⬜ placeholder** means exactly that: the directory holds a README describing what belongs there, and no code. Those six evaluations were run outside this repository and the scripts have not been recovered. Each placeholder README states the benchmark's role, the numbers to reproduce, the upstream harness to drive, and the three checkpoints to run it on.

## Why the second and third groups exist

They are not padding. The paper's central claim needs them.

Agentic post-training buys SWE-Bench points and quietly **charges for them elsewhere** — R2E-Gym alone costs 13.1 points of LiveCodeBench, 7.4 of BFCL, and 4.81 on average across all six. That bill is rarely shown in agent papers. The group-2 and group-3 benchmarks are how the paper shows mid-training pays most of it back (+3.52 average) while still improving the in-domain target.

τ-bench and BFCL carry the most weight of all. They contain no Python code-editing data, and the mid-training corpus contains no tool-use trajectories — so a gain there cannot come from data overlap. It can only come from a structural prior installed at mid-training time that survives post-training, which is the paper's whole thesis about the function-call/tool-call isomorphism.

## Common pattern

Every benchmark scores a checkpoint through an **OpenAI-compatible endpoint**, so the serving step is shared:

```bash
../evaluation/swebench/start_vllm_server.sh /path/to/checkpoint
```

Then drive the benchmark's own official harness against `http://127.0.0.1:$PORT/v1`. Do not reimplement a benchmark — a number only means something if the official harness produced it.

## Protocol

All numbers in the paper are the **mean over three independent evaluation seeds** on the **final checkpoint** of each pipeline. Both arms — baseline and ours — go through the identical harness. Comparing your mid-trained run against a *published* baseline instead of your own reproduction is how a harness difference gets reported as a method gain; the paper reports its own R2E-Gym reproduction (which is below the official number at 7B) precisely to avoid that.
