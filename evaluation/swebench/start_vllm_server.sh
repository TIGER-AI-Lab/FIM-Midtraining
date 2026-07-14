#!/usr/bin/env bash
#
# Stage 1/3 — serve the checkpoint with vLLM.
#
# The agent harness talks to this over an OpenAI-compatible API, so the model
# is just an endpoint as far as the rollout is concerned.
#
#   ./start_vllm_server.sh /path/to/checkpoint
#   MODEL=<hf-org>/<ckpt> PORT=9002 GPUS=0,1 ./start_vllm_server.sh
#
# Leave it running; open a second shell for stage 2.

set -euo pipefail

MODEL="${1:-${MODEL:?set MODEL or pass a checkpoint path as \$1}}"
PORT="${PORT:-9002}"
GPUS="${GPUS:-0}"
SERVED_NAME="${SERVED_NAME:-fim-eval}"

# The agent's trajectories are long: tool outputs (file contents, test logs) are
# appended turn after turn. 64K is what we ran; below ~32K the longer rollouts
# get truncated mid-episode and silently score as failures.
MAX_LEN="${MAX_LEN:-65536}"
TP_SIZE="${TP_SIZE:-1}"

export CUDA_VISIBLE_DEVICES="$GPUS"

echo "model:  $MODEL"
echo "served: $SERVED_NAME  (port $PORT, GPUs $GPUS, TP=$TP_SIZE, max_len=$MAX_LEN)"

vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_LEN" \
  --hf-overrides "{\"max_position_embeddings\": $MAX_LEN}" \
  --enable-prefix-caching \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}"
