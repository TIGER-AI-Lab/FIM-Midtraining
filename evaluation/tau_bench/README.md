# τ-bench

Tool-agent conversations in retail/airline domains, scored on task completion against a user simulator. We report the **unweighted mean of the retail and airline success rates**.

**Role in the paper:** Tool use, fully outside the coding domain. This is one of the paper's two load-bearing transfer results: the mid-training corpus contains NO tool-use trajectories and NO non-Python data, so a gain here can only come from a structural prior installed at mid-training that survives post-training.

## Numbers to reproduce

Paper Table 2 (capability preservation), Qwen2.5-Coder-14B-Instruct + R2E-Gym. All three arms use the same checkpoints; only the benchmark differs.

| Arm | Checkpoint | Score |
|---|---|---|
| Instruct (ceiling) | [Qwen/Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) | 5.70 |
| + R2E-Gym | [R2E-Gym/R2EGym-14B-Agent](https://huggingface.co/R2E-Gym/R2EGym-14B-Agent) | 3.40 |
| **+ FIM Mid-Train + R2E-Gym (ours)** | [TIGER-Lab/FIM-14B](https://huggingface.co/TIGER-Lab/FIM-14B) | **7.30** |

## Reproduce

First serve the checkpoint with vLLM ([`../README.md`](../README.md#common-pattern)) — τ-bench relies on structured tool calls, so the `--enable-auto-tool-choice --tool-call-parser hermes` serving flags are **required**. The commands below assume the endpoint `http://127.0.0.1:8010/v1` with served name `fim-14b`.

```bash
git clone https://github.com/sierra-research/tau-bench && cd tau-bench
pip install -e .
```

The evaluated model plays **both the agent and the user simulator** (no external API key needed):

```bash
export OPENAI_API_BASE=http://127.0.0.1:8010/v1
export OPENAI_BASE_URL=http://127.0.0.1:8010/v1
export OPENAI_API_KEY=EMPTY

for env in retail airline; do
  python run.py --agent-strategy tool-calling --env $env \
    --model fim-14b --model-provider openai \
    --user-model fim-14b --user-model-provider openai \
    --user-strategy llm --task-split test --max-concurrency 4
done
```

Each run prints the average reward (success rate) and writes trajectories under `results/`. The number in the paper's table is the unweighted mean of the retail and airline success rates.

The LLM user simulator carries run-to-run variance.

## Upstream

https://github.com/sierra-research/tau-bench
