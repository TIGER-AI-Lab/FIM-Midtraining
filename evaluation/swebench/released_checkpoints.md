# Scoring the Released Checkpoints on SWE-Bench

The generic three-stage flow is in [`README.md`](README.md). This page pins the exact commands for the three **released** post-trained checkpoints, matching how the paper's headline numbers were produced.

| Checkpoint | Post-training | Scaffold for inference | Scorer |
|---|---|---|---|
| [TIGER-Lab/FIM-7B](https://huggingface.co/TIGER-Lab/FIM-7B) | R2E-Gym | R2E-Gym agent (`edit.py`) | official SWE-bench harness |
| [TIGER-Lab/FIM-14B](https://huggingface.co/TIGER-Lab/FIM-14B) | R2E-Gym | R2E-Gym agent (`edit.py`) | official SWE-bench harness |
| [TIGER-Lab/FIM-8B](https://huggingface.co/TIGER-Lab/FIM-8B) | SWE-Lego | OpenHands `CodeActAgent` | official SWE-bench harness |

The scaffold is **fixed by the post-training pipeline** — evaluating the SWE-Lego model through the R2E-Gym scaffold (or vice versa) produces numbers that are not comparable to the paper's.

## FIM-7B / FIM-14B (R2E-Gym scaffold)

### 1. Serve with vLLM

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
python -m vllm.entrypoints.openai.api_server \
  --model TIGER-Lab/FIM-7B \
  --served-model-name FIM-7B \
  --host 127.0.0.1 \
  --port 8400 \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --hf-overrides '{"max_position_embeddings": 65536}' \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.9
```

For FIM-14B, swap the model and served name. Keep `--max-model-len 65536`: agent trajectories accumulate tool output turn after turn, and below ~32K the long rollouts get truncated mid-episode and silently score as failures.

### 2. Roll out the R2E-Gym agent

From an upstream, unmodified [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym) checkout:

```bash
export OPENAI_API_KEY=EMPTY
export LLM_BASE_URL="http://127.0.0.1:8400/v1"

uv run python src/r2egym/agenthub/run/edit.py runagent_multiple \
  --dataset "R2E-Gym/SWE-Bench-Verified" \
  --split "test" \
  --start_idx 0 \
  --k 500 \
  --traj_dir "./traj" \
  --exp_name "FIM-7B_swebench_verified_r1" \
  --llm_name "openai/FIM-7B" \
  --scaffold "r2egym" \
  --backend "docker" \
  --use_fn_calling False \
  --temperature 0 \
  --max_steps 40 \
  --max_steps_absolute 100 \
  --max_workers 6 \
  --max_reward_calc_time 1200 \
  --max_tokens 65536 \
  --use_existing True
```

For SWE-Bench-Lite use `--dataset "R2E-Gym/SWE-Bench-Lite" --k 300`. Equivalently, use [`run_rollout.sh`](run_rollout.sh) with `PORT=8400 SERVED_NAME=FIM-7B`.

### 3. Score with the official harness

```bash
R2E_GYM=... SWE_BENCH=... ./score.sh verified ./traj/<trajectory>.jsonl
```

## FIM-8B (SWE-Lego / OpenHands scaffold)

### 1. Serve with vLLM

The checkpoint ships `max_position_embeddings: 163840` and its own chat template, so no rope or template overrides are needed:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m vllm.entrypoints.openai.api_server \
  --model TIGER-Lab/FIM-8B \
  --served-model-name FIM-8B \
  --host 127.0.0.1 \
  --port 8400 \
  --tensor-parallel-size 1 \
  --max-model-len 163840 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.9
```

### 2. Run OpenHands `CodeActAgent`

Inference uses OpenHands 0.53.0. Define the LLM in `config.toml`:

```toml
[llm.eval_fim]
model = "openai/FIM-8B"
base_url = "http://127.0.0.1:8400/v1"
api_key = "EMPTY"
temperature = 0.0
max_input_tokens = 147456
max_output_tokens = 16384
native_tool_calling = false
```

From the OpenHands checkout:

```bash
env USE_HINT_TEXT=false \
    INSTRUCTION_TEMPLATE_NAME=swe_default.j2 \
    ENABLE_PLAN_MODE=false \
    ADD_IN_CONTEXT_LEARNING_EXAMPLE=false \
poetry run python evaluation/benchmarks/swe_bench/run_infer.py \
  --config-file config.toml \
  --agent-cls CodeActAgent \
  --llm-config llm.eval_fim \
  --max-iterations 100 \
  --eval-num-workers 1 \
  --eval-output-dir ./eval_out \
  --dataset princeton-nlp/SWE-bench_Verified \
  --split test \
  --mode swe
```

For SWE-Bench Lite, use `--dataset princeton-nlp/SWE-bench_Lite`.

### 3. Score with the official harness

Convert the OpenHands `output.jsonl` to a predictions file with `evaluation/benchmarks/swe_bench/scripts/eval/convert_oh_output_to_swe_json.py`, then evaluate with the official SWE-bench harness (`python -m swebench.harness.run_evaluation`). The reported score is `resolved_instances / total_instances`.

## Protocol

The paper reports the **mean over three independent evaluation seeds** per arm per split. The single greedy pass above (`--temperature 0`) reproduces one seed; the paper's rollout protocol in [`run_rollout.sh`](run_rollout.sh) uses temperature 1.0 so that re-running yields a fresh seed to average.
