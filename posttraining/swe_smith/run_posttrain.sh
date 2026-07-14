#!/usr/bin/env bash
#
# SWE-Smith post-training — one reference launcher.
#
# Run it twice to reproduce the paper's SWE-Smith row-pair. The only thing that
# changes between the two runs is the checkpoint you start from:
#
#   # baseline: the stock instruct model
#   MODEL_DIR=/path/to/Qwen2.5-Coder-7B-Instruct EXP_NAME=swe-smith-baseline ./run_posttrain.sh
#
#   # ours: the FIM mid-trained checkpoint
#   MODEL_DIR=/path/to/fim-midtrain/checkpoint-XXX EXP_NAME=swe-smith-fim ./run_posttrain.sh
#
# Everything is an env var; nothing needs editing in the YAML.

set -euo pipefail

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the checkpoint directory to start from}"
EXP_NAME="${EXP_NAME:-swe-smith-posttrain}"
CONFIG="${CONFIG:-$(dirname "$0")/swe_smith_posttrain.yaml}"
DATA_FILE="${DATA_FILE:-./data/swe_smith_trajectories.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/${EXP_NAME}}"

# Paper Appendix C, Table 8: 8 GPUs x per-device 1 x grad-accum 4 = effective 32.
# Change NUM_GPUS and grad accum together if you want to preserve that.
NUM_GPUS="${NUM_GPUS:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-3}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"

if [[ ! -f "$DATA_FILE" ]]; then
  echo "Training data not found: $DATA_FILE" >&2
  echo "Run:  python download_data.py --output $DATA_FILE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "base model:  $MODEL_DIR"
echo "data:        $DATA_FILE"
echo "output:      $OUTPUT_DIR"
echo "GPUs:        $NUM_GPUS   (effective batch size $((NUM_GPUS * GRAD_ACCUM)))"
echo "lr:          $LR, $EPOCHS epochs"
echo

# torchtune reads the tokenizer from the checkpoint dir alongside the weights,
# so MODEL_DIR must contain vocab.json and merges.txt as well as the shards.
tune run \
  --nnodes 1 \
  --nproc_per_node "$NUM_GPUS" \
  full_finetune_distributed \
  --config "$CONFIG" \
  exp_name="$EXP_NAME" \
  output_dir="$OUTPUT_DIR" \
  tokenizer.path="${MODEL_DIR}/vocab.json" \
  tokenizer.merges_file="${MODEL_DIR}/merges.txt" \
  tokenizer.max_seq_len="$MAX_SEQ_LEN" \
  checkpointer.checkpoint_dir="$MODEL_DIR" \
  checkpointer.output_dir="$OUTPUT_DIR" \
  dataset.data_files="$DATA_FILE" \
  optimizer.lr="$LR" \
  epochs="$EPOCHS" \
  gradient_accumulation_steps="$GRAD_ACCUM" \
  2>&1 | tee "${OUTPUT_DIR}/train.log"
