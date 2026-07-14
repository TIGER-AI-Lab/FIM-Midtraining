# LiveCodeBench

Contamination-free competitive-programming problems, scored by executing the generated program against held-out tests. We use **release_v6, pass@1, greedy decoding**.

**Role in the paper:** Non-agent coding — the regression check. This is where agentic post-training does the most damage (−13.10 vs the Instruct ceiling) and where mid-training recovers the most (+11.10).

## Numbers to reproduce

Paper Table 2 (capability preservation), Qwen2.5-Coder-14B-Instruct + R2E-Gym. All three arms use the same checkpoints; only the benchmark differs.

| Arm | Checkpoint | Score |
|---|---|---|
| Instruct (ceiling) | [Qwen/Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) | 37.20 |
| + R2E-Gym | [R2E-Gym/R2EGym-14B-Agent](https://huggingface.co/R2E-Gym/R2EGym-14B-Agent) | 24.10 |
| **+ FIM Mid-Train + R2E-Gym (ours)** | [TIGER-Lab/FIM-14B](https://huggingface.co/TIGER-Lab/FIM-14B) | **35.20** |

## Reproduce

Unlike the other five benchmarks, LiveCodeBench loads the model directly through its own vLLM runner — **no server needed**.

```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git && cd LiveCodeBench
uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e .
```

If `lcb_runner/lm_styles.py` has no `Qwen2.5-Coder-14B-Instruct` entry, register one (append inside the `LanguageModelList`, next to the existing Qwen2.5-Coder entries):

```python
    LanguageModel(
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        "Qwen2.5-Coder-Ins-14B",
        LMStyle.CodeQwenInstruct,
        datetime(2024, 6, 30),
        link="https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct",
    ),
```

Then run generation + evaluation:

```bash
python -m lcb_runner.runner.main \
  --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --scenario codegeneration --release_version release_v6 \
  --n 1 --temperature 0 --max_tokens 4096 \
  --evaluate --num_process_evaluate 12 --timeout 6
```

For the two fine-tuned checkpoints, keep the same `--model` (they share the Qwen2.5-Coder chat template) and add `--local_model_path /path/to/checkpoint` (a local download of `R2E-Gym/R2EGym-14B-Agent` or `TIGER-Lab/FIM-14B`). Pass@1 is printed at the end and saved under `output/`.

Deterministic under greedy decoding — should reproduce closely.

## Upstream

https://github.com/LiveCodeBench/LiveCodeBench
