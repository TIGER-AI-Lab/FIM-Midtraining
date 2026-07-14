# OJBench

Competitive-programming problems from NOI and ICPC, judged by a real online-judge sandbox (DMOJ judge-server). We use the **full set** and report the **AC rate**.

**Role in the paper:** Non-agent coding — regression check. Mid-training recovers +1.94 of the −2.40 that post-training costs, landing within 0.46 of the Instruct ceiling.

## Numbers to reproduce

Paper Table 2 (capability preservation), Qwen2.5-Coder-14B-Instruct + R2E-Gym. All three arms use the same checkpoints; only the benchmark differs.

| Arm | Checkpoint | Score |
|---|---|---|
| Instruct (ceiling) | [Qwen/Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) | 5.20 |
| + R2E-Gym | [R2E-Gym/R2EGym-14B-Agent](https://huggingface.co/R2E-Gym/R2EGym-14B-Agent) | 2.80 |
| **+ FIM Mid-Train + R2E-Gym (ours)** | [TIGER-Lab/FIM-14B](https://huggingface.co/TIGER-Lab/FIM-14B) | **4.74** |

## Reproduce

First serve the checkpoint with vLLM ([`../README.md`](../README.md#common-pattern)); the commands below assume an OpenAI-compatible endpoint at `http://127.0.0.1:8010/v1` with served name `fim-14b`.

Requires `g++` (C++17), `pypy3`, and `libseccomp-dev` (judge-server compiles its sandbox against `seccomp.h`). Install pypy3 system-wide so the judge sandbox can access it: `apt install pypy3 libseccomp-dev`.

```bash
git clone https://github.com/DMOJ/judge-server.git && cd judge-server
pip install . && cd ..

git clone https://github.com/He-Ren/OJBench.git
# edit OJBench/ojbench/runtime.yaml so g++17/pypy3 point to your binaries
pip install -e OJBench

git lfs install
git clone https://huggingface.co/datasets/He-Ren/OJBench_testdata
```

Generate responses from the served model (OJBench leaves generation to you; any OpenAI client works):

```python
import json, concurrent.futures
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8010/v1", api_key="EMPTY")
prompts = [json.loads(l) for l in open("OJBench_testdata/prompts/full.jsonl")]

def gen(row):
    resp = client.chat.completions.create(
        model="fim-14b",
        messages=[{"role": "user", "content": row["prompt"]}],
        temperature=0, max_tokens=4096)
    return {**row, "content": resp.choices[0].message.content}

with concurrent.futures.ThreadPoolExecutor(8) as pool:
    rows = list(pool.map(gen, prompts))
with open("model_response.jsonl", "w") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
```

Judge and compute the AC rate:

```python
import json
from pathlib import Path
import ojbench

ojbench.init(problem_dirs=[Path("OJBench_testdata/NOI"), Path("OJBench_testdata/ICPC")])
results = ojbench.judge_jsonl("model_response.jsonl", "judged.jsonl", num_workers=16)
print("AC rate:", sum(r["is_passed"] for r in results) / len(results))
```

Note: if the judge fails to resolve numeric problem ids for NOI problems, prefix them with `loj-` (e.g. `1000` → `loj-1000`) before judging.

Deterministic under greedy decoding — should reproduce closely.

## Upstream

https://github.com/He-Ren/OJBench
