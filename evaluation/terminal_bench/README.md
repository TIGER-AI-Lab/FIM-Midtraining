# Terminal-Bench 2.0

Agentic tasks in a real terminal sandbox (89 tasks), run through [Harbor](https://github.com/laude-institute/harbor). We report the **mean reward**. Needs Docker.

**Role in the paper:** Agent OOD — an agent benchmark that is not SWE-Bench. Note the Instruct model scores 0.00 here (it cannot drive an agent loop at all), so both trained arms are improvements over a floor, not a recovery toward a ceiling.

## Numbers to reproduce

Paper Table 2 (capability preservation), Qwen2.5-Coder-14B-Instruct + R2E-Gym. All three arms use the same checkpoints; only the benchmark differs.

| Arm | Checkpoint | Score |
|---|---|---|
| Instruct | [Qwen/Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) | 0.00 |
| + R2E-Gym | [R2E-Gym/R2EGym-14B-Agent](https://huggingface.co/R2E-Gym/R2EGym-14B-Agent) | 2.41 |
| **+ FIM Mid-Train + R2E-Gym (ours)** | [TIGER-Lab/FIM-14B](https://huggingface.co/TIGER-Lab/FIM-14B) | **3.66** |

## Reproduce

First serve the checkpoint with vLLM ([`../README.md`](../README.md#common-pattern)). Agents run **inside containers**, so the endpoint must be reachable from Docker — with the default bridge network that is `http://172.17.0.1:8010/v1`, not `127.0.0.1`.

```bash
pip install harbor

export OPENAI_BASE_URL=http://172.17.0.1:8010/v1
export OPENAI_API_KEY=EMPTY
harbor run --dataset terminal-bench@2.0 \
  --agent qwen-code \
  --model qwen2.5-coder-14b \
  --n-concurrent 4
```

(`--model` must equal the vLLM served name.) Mean reward over the 89 tasks is reported in the job summary.

Live containers + agent scaffold carry run-to-run variance.

## Upstream

https://github.com/laude-institute/terminal-bench · https://github.com/laude-institute/harbor
