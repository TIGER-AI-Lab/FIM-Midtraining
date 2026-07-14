# FullStackBench (English subset)

Multi-domain full-stack programming problems executed in ByteDance's SandboxFusion. We use the **English split** and report the **pass rate**.

**Role in the paper:** Non-agent coding — regression check. Post-training costs −6.08 vs the Instruct ceiling; mid-training claws back +0.53 of it.

## Numbers to reproduce

Paper Table 2 (capability preservation), Qwen2.5-Coder-14B-Instruct + R2E-Gym. All three arms use the same checkpoints; only the benchmark differs.

| Arm | Checkpoint | Score |
|---|---|---|
| Instruct (ceiling) | [Qwen/Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) | 53.80 |
| + R2E-Gym | [R2E-Gym/R2EGym-14B-Agent](https://huggingface.co/R2E-Gym/R2EGym-14B-Agent) | 47.72 |
| **+ FIM Mid-Train + R2E-Gym (ours)** | [TIGER-Lab/FIM-14B](https://huggingface.co/TIGER-Lab/FIM-14B) | **48.25** |

## Reproduce

First serve the checkpoint with vLLM ([`../README.md`](../README.md#common-pattern)); the commands below assume an OpenAI-compatible endpoint at `http://127.0.0.1:8010/v1`.

Docker is required for the execution sandbox:

```bash
docker run -d --rm -p 8080:8080 volcengine/sandbox-fusion:server-20241204

git clone https://github.com/bytedance/FullStackBench.git && cd FullStackBench
pip install -r requirements.txt
```

Edit `src/main.py` to point at the served model — set `client = AsyncOpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8010/v1")`, put the served name in `model=`, and raise `max_tokens` to `4096`. The English split (`./data/fsb_en_20241204.jsonl`) is already the default. Then:

```bash
python src/main.py
```

The pass rate is printed at the end; per-sample results land in `results.jsonl`.

Sandbox execution carries some run-to-run variance.

## Upstream

https://github.com/bytedance/FullStackBench
