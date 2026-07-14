#!/usr/bin/env bash
#
# Stage 2/3 — roll the agent out on SWE-Bench, against the vLLM server.
#
# Each instance runs in its own Docker container: the agent reads files, edits
# them, runs tests, and finally emits a patch. This is the slow, expensive step
# — hours, and it needs a working Docker daemon with room for the SWE-Bench
# images.
#
#   R2E_GYM=/path/to/R2E-Gym ./run_rollout.sh verified
#   R2E_GYM=/path/to/R2E-Gym ./run_rollout.sh lite
#
# Uses UPSTREAM R2E-Gym unmodified. We pass --backend docker and set
# LLM_BASE_URL explicitly rather than relying on any patched default.

set -euo pipefail

SPLIT="${1:-verified}"
R2E_GYM="${R2E_GYM:?set R2E_GYM to your R2E-Gym checkout}"

PORT="${PORT:-9002}"
SERVED_NAME="${SERVED_NAME:-fim-eval}"
EXP_NAME="${EXP_NAME:-fim-eval}"

case "$SPLIT" in
  verified) DATASET="R2E-Gym/SWE-Bench-Verified"; K=500 ;;
  lite)     DATASET="R2E-Gym/SWE-Bench-Lite";     K=300 ;;
  *) echo "Usage: $0 <verified|lite>" >&2; exit 1 ;;
esac

TRAJ_DIR="${TRAJ_DIR:-./traj_${EXP_NAME}_${SPLIT}}"
mkdir -p "$TRAJ_DIR"

# The harness reaches the model through these two. OPENAI_API_KEY must be set to
# something non-empty for the OpenAI client to construct, but vLLM ignores it.
export LLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

echo "dataset:  $DATASET  (k=$K)"
echo "endpoint: $LLM_BASE_URL  (model: $SERVED_NAME)"
echo "traj dir: $TRAJ_DIR"

cd "$R2E_GYM"

python src/r2egym/agenthub/run/edit.py runagent_multiple \
  --traj_dir "$TRAJ_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset "$DATASET" \
  --split test \
  --k "$K" \
  --scaffold r2egym \
  --use_fn_calling False \
  --backend docker \
  --llm_name "openai/${SERVED_NAME}" \
  --temperature "${TEMPERATURE:-1}" \
  --max_workers "${MAX_WORKERS:-10}" \
  --max_steps "${MAX_STEPS:-40}" \
  --max_steps_absolute "${MAX_STEPS_ABSOLUTE:-100}" \
  --max_tokens "${MAX_TOKENS:-65536}" \
  --max_reward_calc_time "${MAX_REWARD_CALC_TIME:-1200}" \
  --condense_history False
