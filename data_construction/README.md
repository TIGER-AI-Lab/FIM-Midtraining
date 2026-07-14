# Dependency-Aware FIM Dataset Construction

Build supervised fine-tuning data that teaches a model to **fill in missing function bodies from the surrounding codebase** — not from a docstring, but from the call sites, the sibling methods, the shared state, the rest of the file.

Two datasets come out of this repository:

| | what gets masked | what the model must learn |
|---|---|---|
| **single-function** | one function body per file | recover a function from its context |
| **multi-function** | 2–3 *coupled* function bodies per file | recover several functions *and keep them consistent with each other* |

The multi-function variant is the interesting one. Real code patches rarely touch one function — they fix a callee and update its caller, or add a method and wire it in. Masking a caller-callee pair together forces the model to invent both ends of an interface contract and make them agree, which is the skill that agentic coding benchmarks (SWE-bench, SWT-bench, Commit-0) actually exercise.

---

## How it works

The core idea is that **not every function is worth masking**. A three-line getter teaches nothing. A function that wraps some obscure vendor API can't be recovered from context by anyone, model or human, so scoring a model against it measures nothing. The pipeline scores every function on two axes and keeps the ones in the sweet spot:

```
Ĥ(v)  intrinsic complexity     — LOC, cyclomatic complexity, AST depth
Î(v)  contextual inferability  — call sites, internal callees, signature,
                                 docstring, sibling methods in the same class

FIM_Score(v) = Ĥ(v) · Î(v) / (Ĥ(v) + Î(v))
```

That product-over-sum peaks when a function is *both* substantial and recoverable, and collapses when it's either trivial (Ĥ→0) or unguessable (Î→0). A one-sided Gaussian then penalises functions whose difficulty runs further ahead of their context than the ceiling allows. For groups of 2–3 functions the same idea is applied at group level, with an added coupling term and with intra-group signals *subtracted* from Î — because once you mask the whole group, the information its members gave each other is gone.

Selection alone isn't enough, so an LLM critic reads every generated completion against the ground truth, scores it on five dimensions, and separately judges whether the function was ever a fair target. Anything it flags as infeasible or low-quality is dropped before it can reach training.

The five steps:

```
step 1  clone repos                              ─┐  shared by both pipelines,
step 2  flatten to one JSON record per .py file  ─┘  run once

step 3  score functions (or groups) and mask the winners
step 4  Gemini completes the mask, then critiques its own completion   ← costs money
step 5  filter on the critic's scores, emit SFT JSONL
```

---

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY='your-key-here'      # get one at https://aistudio.google.com/apikey
```

Then open **`config.yaml`** — it is the only file you need to edit. The two settings that matter:

```yaml
paths:
  repo_csv: data/code_repo_list_968.csv   # which repositories to mine
  work_dir: workdir                       # where everything gets written
```

`work_dir` will hold the cloned repos (tens of GB for the full list), so point it at a disk with room. Relative paths are resolved against `config.yaml`; absolute paths are used as-is.

### Try it on two repos first

Do not point this at 968 repositories on the first run. Make a small CSV, aim a config at it, and watch one pass go through:

```bash
head -3 data/code_repo_list_968.csv > /tmp/mini.csv     # header + 2 repos
cp config.yaml /tmp/mini.yaml                           # then edit repo_csv/work_dir in it

./scripts/run_all.sh single --config /tmp/mini.yaml     # or: CONFIG=/tmp/mini.yaml ./scripts/run_all.sh single
```

Two repos yield roughly a hundred masked functions and cost a few cents. Once that lands, scale up.

---

## Running it

### The whole thing, one process

```bash
./scripts/run_all.sh single     # single-function pipeline, steps 1-5
./scripts/run_all.sh multi      # multi-function pipeline, steps 1-5
./scripts/run_all.sh both       # steps 1-2 once, then both pipelines
```

Fine for a few hundred functions. Far too slow for the full dataset, because step 4 makes two API calls per record and does them one at a time.

### Step 4 at scale

Step 4 is the expensive step — two Gemini calls per record, and hundreds of thousands of records. It is built to be split across many processes. Each shard owns its own checkpoint file, writes after every single record, and skips what it has already done when restarted. **You can kill any shard at any time and lose at most one record.**

```bash
# steps 1-3 first (cheap, single process)
python common/step_1_download_repos.py
python common/step_2_extract_python_files.py
python single_function/step_3_select_functions.py

# then fan step 4 out across the shard count in config.yaml (default: 200)
./scripts/run_step4_parallel.sh single

# ...wait. then:
python single_function/step_5_prepare_sft_data.py
```

Two independent knobs control how hard you hit the API:

- **`sharding.total_shards`** — how many OS processes. Each is isolated; one crashing costs you nothing.
- **`sharding.concurrency`** — how many requests each process keeps in flight (default `1`, which reproduces the original strictly-sequential behaviour).

Total requests in flight ≈ *running shards × concurrency*. Start conservative and raise it until you see rate-limit warnings in the logs, then back off. `llm.wait_seconds` adds a per-worker pause on top.

Logs land in `<work_dir>/logs/step_4_<pipeline>/shard_N.log`. To see what's still alive:

```bash
ps -ef | grep step_4
```

To restart whatever died, just run `run_step4_parallel.sh` again — finished records are skipped, so re-running is cheap and idempotent.

**The multi-function pipeline needs one extra step first.** Its launcher does it for you, but if you're driving step 4 by hand, pre-split the input so 200 workers don't each load the entire dataset into memory:

```bash
python multi_function/step_4_multi_fim_and_critique.py --pre-shard --total-shards 200
```

---

## The output

`<work_dir>/single_function/sft/single_function_fim_sft.jsonl`
`<work_dir>/multi_function/sft/multi_function_fim_sft_pairs.jsonl`
`<work_dir>/multi_function/sft/multi_function_fim_sft_triples.jsonl`

One JSON object per line, in chat-messages format:

```json
{
  "messages": [
    {"role": "user",      "content": "...the FIM prompt, with # <MASKED_FUNCTION_BODY> in the code..."},
    {"role": "assistant", "content": "### Reasoning\n...\n### Implementation\n```python\n...\n```"}
  ],
  "metadata": {"repo_id": "0", "license": "MIT License", "fim_score": 0.31, "overall_score": 5, "...": "..."}
}
```

The user turn is **byte-identical to the prompt step 4 actually sent**, so the model trains on exactly the distribution it was scored on. The assistant turn is the raw response: reasoning *and* implementation, because the reasoning is the point — you are distilling the act of inferring code from context, not just the code.

`metadata` is not consumed by any trainer. It carries the repo provenance, the selection scores (`fim_score`, `complexity`, `inferability`, `difficulty`) and the critic's scores, so you can re-filter or run ablations without regenerating anything. Strip it freely.

Pairs and triples are written separately so you can weight them differently, or drop triples entirely.

---

## Quality filters

Step 5 is where you trade dataset size against dataset quality. Every threshold lives in `config.yaml`:

```yaml
filters:
  single:
    min_individual_score: 3    # every dimension: correctness, executability,
    min_overall_score: 4       #   api_usage, readability, completeness
  multi:
    min_per_func_score: 3      # EVERY function in the group must clear this
    min_group_overall_score: 4
    min_coherence_score: 3     # cross-function interface/state/logic consistency
```

For groups the rule is deliberately harsh: **one bad function discards the whole group.** A sample where function A correctly calls a broken function B is worse than no sample at all — it teaches the model to write code against an interface that doesn't work.

Step 5 is cheap and re-runnable. Step 4's checkpoints keep the critic's full reasoning, so you can re-filter at different thresholds as many times as you like without spending another cent.

---

## Bringing your own repositories

Replace `paths.repo_csv` with your own CSV. Required columns:

| column | required | what it does |
|---|---|---|
| `sample_id` | **yes** | identifies the repo. Becomes the clone folder name and rides along as `repo_id` in every record. |
| `repository_url` | **yes** | any git-cloneable URL. |
| `category` | no | free text, copied into metadata. |
| `description` | no | free text, copied into metadata. |
| `notes` | no | free text, copied into metadata. |
| `license` | no | free text, copied into metadata. Worth filling in — it's what tells you whether you may train on the result. |

A missing optional column comes through as an empty string; nothing breaks.

```csv
sample_id,repository_url,category,description,notes,license
0,https://github.com/psf/requests,http,HTTP library,,Apache License 2.0
1,https://github.com/pallets/flask,web,Web framework,,BSD 3-Clause
```

> **A note on the two ids.** Your CSV's `sample_id` identifies a *repository*. Step 2 mints a **different** `sample_id` for every source file, and everything downstream keys a FIM record on `(sample_id, func_name, start_line)`. So on ingest the CSV column is carried as `repo_id`. If the repo's id were allowed into the per-file field, every file in a repo would share an id and distinct functions would silently collide in step 4's checkpoint dedup — you'd process a fraction of your data and never see an error. If you rename columns, keep those two ids separate.

The shipped `data/code_repo_list_968.csv` lists 968 permissively-licensed Python repositories with their licenses. **Check the license column against your intended use before training on anything derived from it** — the pipeline copies the license into every sample's metadata precisely so this stays auditable.

---

## Tuning what gets selected

`config.yaml`'s `selection` block controls step 3. The defaults are the ones used to build the released dataset:

```yaml
selection:
  min_file_lines: 50       # a file needs enough context to infer from
  max_file_lines: 1800     # ...but must still fit in a prompt
  min_loc: 10              # skip trivial getters
  max_loc: 200             # skip monsters
  score_threshold: 0.08    # the FIM_Score bar
  min_complexity: 0.15
```

Raise `score_threshold` for a smaller, harder, more expensive-per-sample dataset; lower it for more coverage and more noise. Step 3 prints a full distribution (`loc`, `complexity`, `inferability`, `fim_score`, `difficulty` — count/mean/std/p10-p90) every run, so you can see exactly what a threshold change did before you spend anything on step 4.

Everything not listed in `selection` falls back to the defaults in `common/dep_graph.py`, where the scoring weights (`w_loc`, `alpha_caller`, `difficulty_ceiling`, …) also live.

---

## Cost, and how not to be surprised by it

Step 4 makes **two API calls per record**: one to complete, one to critique. Nothing else in the pipeline touches the network.

Step 4 prints a running cost estimate and every shard's checkpoint stores its token usage. The estimate is computed from `llm.price_per_1m_input_tokens` / `price_per_1m_output_tokens` in `config.yaml` — **these are just numbers for the arithmetic, set them to your actual billing tier.** They do not affect the data.

Before committing to a full run, do a real one on a slice and read the number off:

```bash
python single_function/step_4_fim_and_critique.py --start-idx 0 --end-idx 20 --print-response
```

Twenty records, every prompt and response printed, a cost figure at the end. Multiply.

---

## Layout

```
config.yaml                  ← the only file you need to edit
data/
  code_repo_list_968.csv     968 permissively-licensed Python repos
common/
  config.py                  config loading + all derived paths
  dep_graph.py               the dependency graph and FIM scoring core
  llm_client.py              Gemini client, retries, token/cost accounting
  step_1_download_repos.py   ─┐ shared by both pipelines
  step_2_extract_python_files.py ─┘
single_function/
  step_3_select_functions.py
  step_4_fim_and_critique.py
  step_5_prepare_sft_data.py
multi_function/
  step_3_select_function_groups.py
  step_4_multi_fim_and_critique.py
  step_5_prepare_sft_data.py
scripts/
  run_all.sh                 whole pipeline, one process
  run_step4_parallel.sh      fan step 4 out across shards
```

Every step script also takes `--help` and accepts CLI overrides for anything in `config.yaml`; the flags always win over the file.

---

## Notes

- **Python 3.9+.** No GPU, no local model — step 4 is API calls, everything else is `ast` parsing.
- **Which API key.** The Gemini SDK reads `GEMINI_API_KEY` by preference and `GOOGLE_API_KEY` as a fallback; the client accepts either and logs which one it used at startup. If both are set, `GEMINI_API_KEY` wins.
- **Experiment tracking is optional.** Set `wandb.enabled: true` in `config.yaml` (or pass `--wandb`) after `pip install wandb weave`. Without them installed, the pipelines log a warning and carry on.
- **Only Python source is supported.** The dependency graph is built with Python's own `ast` module. Another language means writing another `DependencyGraphBuilder`; the scoring above it is language-agnostic.
