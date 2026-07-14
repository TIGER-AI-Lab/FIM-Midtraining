# Function-Aware Fill-in-the-Middle as Mid-Training for Coding Agent Foundation Models

[📄 Paper](paper.pdf) · 🤗 Dataset: `[TODO: link]` · 🤗 Models: `[TODO: link]`

A coding agent's inner loop — **act → observe → continue** — is structurally the same shape as a function call site: a caller binds arguments, a callee returns a value computed elsewhere, and downstream code consumes that value. That conditioning structure already exists, at internet scale, in ordinary code.

This repo exploits it. We mask *functions* — chosen by program-dependency-graph analysis and a complexity–inferability double criterion, not at random — and mid-train on recovering them. Then we hand the checkpoint to an existing agentic post-training pipeline, **unmodified**.

The result survives that post-training:

| Base model | Post-training | SWE-Bench-Verified | SWE-Bench-Lite |
|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct | R2E-Gym | 15.00 → **17.80** (+2.8) | 11.33 → **15.00** (+3.7) |
| Qwen2.5-Coder-7B-Instruct | SWE-Smith | 12.30 → **17.60** (+5.3) | 14.20 → **14.70** (+0.5) |
| Qwen2.5-Coder-14B-Instruct | R2E-Gym | 26.20 → **29.20** (+3.0) | 18.00 → **22.00** (+4.0) |
| Qwen3-8B | SWE-Lego | 31.80 → **35.00** (+3.2) | 27.30 → **32.70** (+5.4) |

And it pays back most of the hidden bill that agentic post-training charges elsewhere. R2E-Gym alone costs 13.1 points of LiveCodeBench and 7.4 of BFCL relative to the instruct model — a cost rarely reported in agent papers. Mid-training first restores most of it (+3.52 average over six benchmarks) *while* improving the in-domain target.

The load-bearing result is τ-bench (+3.9) and BFCL (+2.4): neither contains Python code-editing data, and our corpus contains no tool-use trajectories. There is no data overlap that could explain those gains — only a structural prior installed at mid-training that outlives post-training.

---

## The pipeline

```
data_construction/   968 GitHub repos -> ~400K FIM samples (~2.6B tokens)
        |
midtraining/         base model + FIM corpus -> mid-trained checkpoint
        |
posttraining/        + R2E-Gym | SWE-Smith | SWE-Lego  (upstream recipes, unmodified)
        |
evaluation/          SWE-Bench, and six benchmarks that check what it cost you
```

Every stage is run **twice** — once from the stock instruct model (baseline), once from the mid-trained checkpoint (ours). The difference between those two runs is the entire claim. Both arms must go through an identical post-training and evaluation harness; comparing against a *published* baseline instead of your own reproduction is how a harness difference gets reported as a method gain.

| Stage | What's here |
|---|---|
| **[`data_construction/`](data_construction)** | Complete and tested. Builds the corpus end to end from a repo list. |
| **[`midtraining/`](midtraining)** | One LLaMA-Factory reference config. The same recipe is used for all three base models. |
| **[`posttraining/`](posttraining)** | One reference config per pipeline (R2E-Gym, SWE-Smith, SWE-Lego). |
| **[`evaluation/`](evaluation)** | SWE-Bench Verified/Lite complete. The other six benchmarks are **placeholders** — see below. |

## What is not here yet

This repo is explicit about its gaps rather than shipping code that looks complete and isn't:

- **Six of the eight evaluations** (LiveCodeBench, OJBench, FullStackBench, Terminal-Bench, τ-bench, BFCL) have no runner. Each directory holds a README stating the benchmark's role, the numbers to reproduce, the upstream harness to drive, and the three checkpoints to run it on.
- **The corpus mixing step** (80% single / 15% pair / 5% triple) — the three splits are produced, the mixing is not. See [`data_construction/mixing/`](data_construction/mixing).

## Quickstart

```bash
# 1. Build a corpus (start small — two repos, a few cents of API spend)
cd data_construction
pip install -r requirements.txt
export GEMINI_API_KEY='...'
head -3 data/code_repo_list_968.csv > /tmp/mini.csv   # then point config.yaml at it
./scripts/run_all.sh single

# 2. Mid-train  (8x H100, LLaMA-Factory)
#    -> midtraining/README.md

# 3. Post-train twice: from the stock model, and from the mid-trained checkpoint
#    -> posttraining/README.md

# 4. Evaluate both arms through the identical harness
#    -> evaluation/swebench/README.md
```

Start with [`data_construction/README.md`](data_construction/README.md) — it is the one part you can run today, end to end, on a laptop.

## Compute

Everything was run on a **single node of 8x H100 80GB**. Mid-training is 1 epoch at 32K context. SWE-Bench evaluation is by far the slowest step: hundreds of Docker containers per pass, and the paper averages three seeds per arm per split.

## Corpus

968 permissively-licensed Python repositories across 10 topic categories → ~78K self-contained files → ~400K FIM samples (~2.6B tokens under the Qwen2.5-Coder tokenizer): ~320K single-function, ~60K pairs, ~20K triples. Every target carries a Gemini-3-Flash rationale. Zero overlap with SWE-Bench source repositories.

The per-repository license ships alongside the list ([`data_construction/data/code_repo_list_968.csv`](data_construction/data/code_repo_list_968.csv)) and is copied into every sample's metadata, so downstream use stays auditable. **Check it against your intended use before training on anything derived from it.**

## Citation

```bibtex
[TODO: bibtex]
```
