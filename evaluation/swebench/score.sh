#!/usr/bin/env bash
#
# Stage 3/3 — convert the trajectories to a SWE-Bench submission, then score it
# with the OFFICIAL SWE-Bench harness.
#
# The score that goes in the paper comes from here, not from R2E-Gym's internal
# reward — the harness re-runs each repo's real test suite against the patch.
#
#   R2E_GYM=... SWE_BENCH=... ./score.sh verified ./traj_fim-eval_verified/<file>.jsonl
#
# Docker is required again: one container per instance to run the tests.

set -euo pipefail

SPLIT="${1:-verified}"
TRAJ_FILE="${2:?pass the trajectory .jsonl emitted by run_rollout.sh}"

R2E_GYM="${R2E_GYM:?set R2E_GYM to your R2E-Gym checkout}"
SWE_BENCH="${SWE_BENCH:?set SWE_BENCH to your SWE-bench checkout}"

EXP_NAME="${EXP_NAME:-fim-eval}"

case "$SPLIT" in
  verified) DATASET_NAME="princeton-nlp/SWE-bench_Verified"; RUN_ID="${RUN_ID:-swebench_verified}" ;;
  lite)     DATASET_NAME="princeton-nlp/SWE-bench_Lite";     RUN_ID="${RUN_ID:-swebench_lite}" ;;
  *) echo "Usage: $0 <verified|lite> <traj.jsonl>" >&2; exit 1 ;;
esac

RESULTS_DIR="${RESULTS_DIR:-./results}"
mkdir -p "$RESULTS_DIR"
SUBMISSION="$(cd "$RESULTS_DIR" && pwd)/${EXP_NAME}_${SPLIT}_submission.json"

# --- trajectories -> SWE-Bench submission format ---
echo "converting $TRAJ_FILE -> $SUBMISSION"
(
  cd "$R2E_GYM"
  python src/r2egym/agenthub/trajectory/create_swebench_submission.py \
    --traj_file_path "$TRAJ_FILE" \
    --output_json_path "$SUBMISSION"
)

# --- official harness ---
# Note the run_id is distinct per split. Sharing one run_id across Verified and
# Lite makes the harness reuse the previous run's cached results and silently
# report the wrong numbers.
echo "scoring with the official SWE-bench harness ($DATASET_NAME, run_id=$RUN_ID)"
(
  cd "$SWE_BENCH"
  python -m swebench.harness.run_evaluation \
    --dataset_name "$DATASET_NAME" \
    --predictions_path "$SUBMISSION" \
    --max_workers "${MAX_WORKERS:-32}" \
    --run_id "$RUN_ID" \
    --cache_level none
)

echo
echo "done. The resolved rate is in the report JSON the harness wrote to \$SWE_BENCH."
