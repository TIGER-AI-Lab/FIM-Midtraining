#!/usr/bin/env bash
#
# Run a whole pipeline end to end, single process.
#
#   ./scripts/run_all.sh single
#   ./scripts/run_all.sh multi
#   ./scripts/run_all.sh both
#
# Steps 1 and 2 are shared: they clone the repos and flatten them into one JSON
# file. Running `both` does them once and then runs each pipeline's 3/4/5.
#
# This runs step 4 in ONE process. That is fine for a few hundred functions and
# far too slow for the full dataset — for that, use scripts/run_step4_parallel.sh
# to fan it out across shards, then run step 5 by hand. See the README.
#
# Everything reads config.yaml. Override it with CONFIG=/path/to/other.yaml.

set -euo pipefail

RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RELEASE_ROOT"

CONFIG="${CONFIG:-$RELEASE_ROOT/config.yaml}"
PY="${PYTHON:-python3}"
PIPELINE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    single|multi|both) PIPELINE="$1"; shift ;;
    --config)          CONFIG="$2"; shift 2 ;;
    --config=*)        CONFIG="${1#*=}"; shift ;;
    *) echo "Usage: $0 <single|multi|both> [--config path/to/config.yaml]" >&2; exit 1 ;;
  esac
done

if [[ -z "$PIPELINE" ]]; then
  echo "Usage: $0 <single|multi|both> [--config path/to/config.yaml]" >&2
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Error: config not found: $CONFIG" >&2
  exit 1
fi

if [[ -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "Error: set GEMINI_API_KEY (or GOOGLE_API_KEY) before running step 4." >&2
  echo "  export GEMINI_API_KEY='your-key-here'" >&2
  exit 1
fi

banner() { echo; echo "=============================================================="; echo "  $1"; echo "=============================================================="; }

# ---- shared stages -----------------------------------------------------------
banner "Step 1 — cloning repositories"
$PY common/step_1_download_repos.py --config "$CONFIG"

banner "Step 2 — extracting Python files"
$PY common/step_2_extract_python_files.py --config "$CONFIG"

# ---- single-function ---------------------------------------------------------
if [[ "$PIPELINE" == "single" || "$PIPELINE" == "both" ]]; then
  banner "Step 3 (single) — selecting functions to mask"
  $PY single_function/step_3_select_functions.py --config "$CONFIG" --quiet

  banner "Step 4 (single) — completion + critique  [this one costs money]"
  $PY single_function/step_4_fim_and_critique.py --config "$CONFIG"

  banner "Step 5 (single) — filtering + SFT formatting"
  $PY single_function/step_5_prepare_sft_data.py --config "$CONFIG"
fi

# ---- multi-function ----------------------------------------------------------
if [[ "$PIPELINE" == "multi" || "$PIPELINE" == "both" ]]; then
  banner "Step 3 (multi) — selecting coupled function groups"
  $PY multi_function/step_3_select_function_groups.py --config "$CONFIG" --quiet

  banner "Step 4 (multi) — group completion + critique  [this one costs money]"
  $PY multi_function/step_4_multi_fim_and_critique.py --config "$CONFIG"

  banner "Step 5 (multi) — filtering + SFT formatting"
  $PY multi_function/step_5_prepare_sft_data.py --config "$CONFIG"
fi

banner "Done"
