#!/bin/bash

set -euo pipefail

export HF_DATASETS_OFFLINE=1
export HF_HOME="/gemini/space/zyf"

MODEL="/gemini/space/zyf/models/Qwen/Qwen3.5-9B"
TASKS="videomme"
LIMIT="${LIMIT:-}"
BATCH_SIZE=4
OUTPUT_PATH="./logs/qwen35_9b_videomme"
VERBOSITY="${VERBOSITY:-DEBUG}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

echo "[INFO] Local test"
echo "[INFO] model=${MODEL} tasks=${TASKS} limit=${LIMIT} batch_size=${BATCH_SIZE} device_map=${DEVICE_MAP}"
echo "[INFO] output_path=${OUTPUT_PATH}"

# source /gemini/space/zyf/lmms-eval/.oldvenv/bin/activate
python -m lmms_eval \
  --model qwen3_5 \
  --model_args "pretrained=${MODEL},device_map=${DEVICE_MAP}" \
  --tasks "${TASKS}" \
  --batch_size "${BATCH_SIZE}" \
  --output_path "${OUTPUT_PATH}" \
  --log_samples \
  --verbosity "${VERBOSITY}" 
  # --limit 5
