#!/usr/bin/env bash
#
# Fan step 4 out across many processes.
#
# Step 4 is the only step that costs money and the only one that takes days
# rather than minutes. Each shard is an independent process with its own
# checkpoint, so you can kill any of them, lose at most one record, and relaunch
# — already-processed records are skipped on restart.
#
#   ./scripts/run_step4_parallel.sh single
#   ./scripts/run_step4_parallel.sh multi
#   SHARD_START=51 SHARD_END=100 ./scripts/run_step4_parallel.sh single
#
# Shard count, launch range and per-process concurrency all come from
# config.yaml (the `sharding` block); the env vars above override them.
#
# Logs land in <work_dir>/logs/. To see what is still running:
#   ps -ef | grep step_4
# To relaunch whatever died, just run this script again — finished records are
# skipped, so re-running is cheap and safe.

set -euo pipefail

RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RELEASE_ROOT"

CONFIG="${CONFIG:-$RELEASE_ROOT/config.yaml}"
PIPELINE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    single|multi) PIPELINE="$1"; shift ;;
    --config)     CONFIG="$2"; shift 2 ;;
    --config=*)   CONFIG="${1#*=}"; shift ;;
    *) echo "Usage: $0 <single|multi> [--config path/to/config.yaml]" >&2; exit 1 ;;
  esac
done

if [[ -z "$PIPELINE" ]]; then
  echo "Usage: $0 <single|multi> [--config path/to/config.yaml]" >&2
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Error: config not found: $CONFIG" >&2
  exit 1
fi

# Read the sharding block + work_dir out of config.yaml.
read -r TOTAL_SHARDS CFG_START CFG_END CFG_CONCURRENCY WORK_DIR <<< "$(
  python3 - "$CONFIG" <<'PY'
import sys
sys.path.insert(0, ".")
from common.config import derive_paths, load_config
cfg = load_config(sys.argv[1])
s = cfg["sharding"]
paths = derive_paths(cfg)
print(s["total_shards"], s["shard_start"], s["shard_end"], s["concurrency"], paths["work_dir"])
PY
)"

# Env vars win over config.yaml.
SHARD_START="${SHARD_START:-$CFG_START}"
SHARD_END="${SHARD_END:-$CFG_END}"
CONCURRENCY="${CONCURRENCY:-$CFG_CONCURRENCY}"

# Seconds to wait between launching processes. Staggering matters: 200 processes
# all hitting the API in the same second is a good way to get rate-limited.
STAGGER="${STAGGER:-10}"

if [[ "$PIPELINE" == "single" ]]; then
  SCRIPT="single_function/step_4_fim_and_critique.py"
else
  SCRIPT="multi_function/step_4_multi_fim_and_critique.py"
fi

LOG_DIR="$WORK_DIR/logs/step_4_$PIPELINE"
mkdir -p "$LOG_DIR"

echo "Pipeline:     $PIPELINE"
echo "Script:       $SCRIPT"
echo "Shards:       $SHARD_START..$SHARD_END of $TOTAL_SHARDS"
echo "Concurrency:  $CONCURRENCY requests in flight per process"
echo "Logs:         $LOG_DIR"
echo

# The multi-function pipeline pre-splits its input once, so each worker reads
# only its own small shard file instead of loading the whole dataset.
if [[ "$PIPELINE" == "multi" ]]; then
  echo "Pre-sharding the input (one-off)..."
  python3 "$SCRIPT" --config "$CONFIG" --pre-shard --total-shards "$TOTAL_SHARDS"
  echo
fi

for i in $(seq "$SHARD_START" "$SHARD_END"); do
  nohup python3 "$SCRIPT" \
    --config "$CONFIG" \
    --shard "$i" --total-shards "$TOTAL_SHARDS" \
    --concurrency "$CONCURRENCY" \
    > "$LOG_DIR/shard_$i.log" 2>&1 &
  echo "launched shard $i (pid $!)"
  sleep "$STAGGER"
done

echo
echo "All shards launched. Watch progress with:"
echo "  tail -f $LOG_DIR/shard_${SHARD_START}.log"
