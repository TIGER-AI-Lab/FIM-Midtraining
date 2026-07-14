# Evaluation

Eight benchmarks in three groups. Every one of them scores a **post-trained** checkpoint — the paper deliberately never evaluates a mid-training-only model, because a FIM-only model has degraded instruction-following and could not be compared fairly against an instruction-tuned baseline. Every reported gain is one that survived post-training.

| Group | Benchmark | Paper table | Status |
|---|---|---|---|
| **Coding agent** (in-domain) | [SWE-Bench-Verified](swebench) | Table 1 | ✅ scripts here |
| | [SWE-Bench-Lite](swebench) | Table 1 | ✅ scripts here |
| **Non-agent coding** (regression check) | [LiveCodeBench](livecodebench) | Table 2 | ✅ repro guide |
| | [OJBench](ojbench) | Table 2 | ✅ repro guide |
| | [FullStackBench-EN](fullstackbench) | Table 2 | ✅ repro guide |
| **Agent OOD** | [Terminal-Bench 2.0](terminal_bench) | Table 2 | ✅ repro guide |
| **Tool use** (outside coding) | [τ-bench](tau_bench) | Table 2 | ✅ repro guide |
| | [BFCL](bfcl) | Table 2 | ✅ repro guide |

Each Table-2 directory documents the exact reproduction path: which upstream harness at which version, how to point it at a served checkpoint, and the three arms to run — the instruct base ([Qwen/Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct)), the post-trained-only arm ([R2E-Gym/R2EGym-14B-Agent](https://huggingface.co/R2E-Gym/R2EGym-14B-Agent)), and ours ([TIGER-Lab/FIM-14B](https://huggingface.co/TIGER-Lab/FIM-14B)). All six were run at 14B; scores are in each directory's README and in the paper's Table 2.

## Why the second and third groups exist

They are not padding. The paper's central claim needs them.

Agentic post-training buys SWE-Bench points and quietly **charges for them elsewhere** — R2E-Gym alone costs 13.1 points of LiveCodeBench, 7.4 of BFCL, and 4.81 on average across all six. That bill is rarely shown in agent papers. The group-2 and group-3 benchmarks are how the paper shows mid-training pays most of it back (+3.52 average) while still improving the in-domain target.

τ-bench and BFCL carry the most weight of all. They contain no Python code-editing data, and the mid-training corpus contains no tool-use trajectories — so a gain there cannot come from data overlap. It can only come from a structural prior installed at mid-training time that survives post-training, which is the paper's whole thesis about the function-call/tool-call isomorphism.

## Common pattern

Every benchmark except LiveCodeBench (which loads the model itself) scores a checkpoint through an **OpenAI-compatible endpoint**, so the serving step is shared. For SWE-Bench use [`swebench/start_vllm_server.sh`](swebench/start_vllm_server.sh); for the six Table-2 benchmarks serve at 32K context:

```bash
vllm serve TIGER-Lab/FIM-14B \
  --served-model-name fim-14b \
  --host 0.0.0.0 --port 8010 \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --trust-remote-code
```

Swap the model and served name per arm (`Qwen/Qwen2.5-Coder-14B-Instruct` → `qwen2.5-coder-14b`, `R2E-Gym/R2EGym-14B-Agent` → `r2egym-model`). The `--enable-auto-tool-choice --tool-call-parser hermes` flags are **required** for τ-bench and BFCL (they rely on structured tool calls; Qwen2.5 emits Hermes-style tool calls) and harmless for the rest. The endpoint is `http://127.0.0.1:8010/v1` with API key `EMPTY`.

Then drive the benchmark's own official harness against that endpoint. Do not reimplement a benchmark — a number only means something if the official harness produced it. Unless a harness exposes otherwise, all generations use temperature 0 and `max_tokens=4096`; table values are rounded to one decimal place.

## Protocol

All numbers in the paper are the **mean over three independent evaluation seeds** on the **final checkpoint** of each pipeline. Both arms — baseline and ours — go through the identical harness. Comparing your mid-trained run against a *published* baseline instead of your own reproduction is how a harness difference gets reported as a method gain; the paper reports its own R2E-Gym reproduction (which is below the official number at 7B) precisely to avoid that.
