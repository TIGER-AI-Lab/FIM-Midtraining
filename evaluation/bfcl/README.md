# BFCL (Berkeley Function-Calling Leaderboard)

Function-calling accuracy across single/multiple/parallel calls, AST-matched and executed. We use **`all_scoring`** and report the **overall accuracy**.

**Role in the paper:** Tool use, outside the coding domain. The other load-bearing transfer result — and the most direct probe of the paper's function-call/tool-call isomorphism claim.

## Numbers to reproduce

Paper Table 2 (capability preservation), Qwen2.5-Coder-14B-Instruct + R2E-Gym. All three arms use the same checkpoints; only the benchmark differs.

| Arm | Checkpoint | Score |
|---|---|---|
| Instruct (ceiling) | [Qwen/Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) | 23.20 |
| + R2E-Gym | [R2E-Gym/R2EGym-14B-Agent](https://huggingface.co/R2E-Gym/R2EGym-14B-Agent) | 15.80 |
| **+ FIM Mid-Train + R2E-Gym (ours)** | [TIGER-Lab/FIM-14B](https://huggingface.co/TIGER-Lab/FIM-14B) | **18.20** |

## Reproduce

First serve the checkpoint with vLLM ([`../README.md`](../README.md#common-pattern)) — BFCL relies on structured tool calls, so the `--enable-auto-tool-choice --tool-call-parser hermes` serving flags are **required**. The commands below assume the endpoint `http://127.0.0.1:8010/v1` with served name `qwen2.5-coder-14b` (swap per arm).

```bash
git clone https://github.com/ShishirPatil/gorilla.git && cd gorilla
git checkout 6ea5797
pip install -e berkeley-function-call-leaderboard
pip install soundfile   # unlisted transitive dep: model_config imports qwen_agent, which needs it
```

The pinned commit has no entry for these models, so register one that routes through the OpenAI-compatible endpoint — append to `berkeley-function-call-leaderboard/bfcl_eval/constants/model_config.py`:

```python
api_inference_model_map["qwen2.5-coder-14b-FC"] = ModelConfig(
    model_name="qwen2.5-coder-14b",   # must equal the vLLM served name
    display_name="qwen2.5-coder-14b-FC",
    url="http://localhost/openai-compatible",
    org="local",
    license="unknown",
    model_handler=OpenAICompletionsHandler,
    input_price=None,
    output_price=None,
    is_fc_model=True,
    underscore_to_dot=True,
)
MODEL_CONFIG_MAPPING["qwen2.5-coder-14b-FC"] = api_inference_model_map["qwen2.5-coder-14b-FC"]
```

Then generate and evaluate. The `OpenAICompletionsHandler` registered above reads `OPENAI_BASE_URL` / `OPENAI_API_KEY` (the `REMOTE_OPENAI_*` variables only apply to the local/OSS handler path), so point those at the vLLM endpoint:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8010/v1
export OPENAI_API_KEY=EMPTY

bfcl generate --model qwen2.5-coder-14b-FC --test-category all_scoring --num-threads 4
bfcl evaluate --model qwen2.5-coder-14b-FC --test-category all_scoring
```

The overall accuracy is reported in the evaluation output (`score/data_overall.csv`).

Deterministic under greedy decoding — should reproduce closely.

## Upstream

https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
