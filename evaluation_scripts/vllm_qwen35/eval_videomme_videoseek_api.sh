#!/bin/bash

# ============================================================
# VideoMME Evaluation Script via VideoSeek + vLLM API
# ============================================================
# This script evaluates VideoSeek on VideoMME. VideoSeek calls
# a vLLM/OpenAI-compatible API server through LiteLLM.
# ============================================================

set -euo pipefail

# ----------------------
# Configuration
# ----------------------
PORT="${1:-5590}"
HOST="10.233.45.74"
API_BASE="http://${HOST}:${PORT}/v1"
API_KEY="any"
MODEL_VERSION="Qwen3.5-397B-A17B-FP8"
VIDEOSEEK_MODEL_VERSION="openai/${MODEL_VERSION}"
VIDEOSEEK_ROOT="/gemini/space/gjx/videoseek"

# Task Configuration
export HF_DATASETS_OFFLINE=1
export HF_HOME="/gemini/space/zyf"
TASKS="videomme"
DATASET_PATH="/gemini/space/zyf/datasets/lmms-lab/Video-MME"

# Evaluation Configuration
BATCH_SIZE=16
NUM_WORKERS="${NUM_WORKERS:-8}"
LIMIT="${LIMIT:-}"
OUTPUT_PATH="./logs/videoseek_qwen35_397b_videomme_api"
LOG_SUFFIX="videoseek_qwen35_397b_api_$(date +%Y%m%d_%H%M%S)"
VERBOSITY="${VERBOSITY:-DEBUG}"

# VideoSeek args
MAX_STEPS="${MAX_STEPS:-10}"
MAX_TOKENS="${MAX_TOKENS:-102400}"
REASONING_EFFORT="${REASONING_EFFORT:-none}"
TEMPERATURE="${TEMPERATURE:-0}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
ACTION_PARSE_MODE="${ACTION_PARSE_MODE:-tool_call}"

# ----------------------
# Check Server Health
# ----------------------
echo "=============================================="
echo "VideoMME Evaluation via VideoSeek"
echo "=============================================="
echo ""
echo "Checking server health..."

if ! curl -s "${API_BASE}/../health" >/dev/null 2>&1; then
    echo "Warning: health endpoint is not available at ${API_BASE}/../health"
    echo "Continuing anyway because some OpenAI-compatible services do not expose /health."
fi

echo ""

# ----------------------
# Print Configuration
# ----------------------
echo "Configuration:"
echo "  API Base URL:  ${API_BASE}"
echo "  Model:         ${VIDEOSEEK_MODEL_VERSION}"
echo "  VideoSeek:     ${VIDEOSEEK_ROOT}"
echo "  Tasks:         ${TASKS}"
echo "  Batch Size:    ${BATCH_SIZE}"
echo "  Workers:       ${NUM_WORKERS}"
echo "  Dataset:       ${DATASET_PATH}"
echo "  Output Path:   ${OUTPUT_PATH}"
echo "  Limit:         ${LIMIT:-<none>}"
echo "  Action Parser: ${ACTION_PARSE_MODE}"
echo ""
echo "=============================================="
echo ""

if [ ! -d "${VIDEOSEEK_ROOT}" ]; then
    echo "Error: VideoSeek repo not found: ${VIDEOSEEK_ROOT}"
    exit 1
fi

# ----------------------
# Run Evaluation
# ----------------------
python -m lmms_eval \
  --model videoseek \
  --force_simple \
  --model_args "videoseek_root=${VIDEOSEEK_ROOT},model_name=${VIDEOSEEK_MODEL_VERSION},api_base=${API_BASE},api_key=${API_KEY},max_steps=${MAX_STEPS},max_tokens=${MAX_TOKENS},reasoning_effort=${REASONING_EFFORT},action_parse_mode=${ACTION_PARSE_MODE},temperature=${TEMPERATURE},timeout=${REQUEST_TIMEOUT},num_workers=${NUM_WORKERS}" \
  --tasks "${TASKS}" \
  --batch_size "${BATCH_SIZE}" \
  ${LIMIT} \
  --log_samples \
  --log_samples_suffix "${LOG_SUFFIX}" \
  --output_path "${OUTPUT_PATH}" \
  --verbosity "${VERBOSITY}"

echo ""
echo "=============================================="
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_PATH}"
echo "=============================================="
