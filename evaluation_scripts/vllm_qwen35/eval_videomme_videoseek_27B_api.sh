#!/bin/bash

# ============================================================
# VideoMME Evaluation Script via VideoSeek + vLLM API
# ============================================================
# This script evaluates VideoSeek on VideoMME using the
# vLLM OpenAI-compatible API server backed by Qwen3.5-27B.
#
# Prerequisites:
#   1. Start the vLLM server for Qwen3.5-27B
#   2. Wait for the server to be ready
#   3. Run this script
#
# Usage:
#   bash eval_videomme_videoseek_27B_api.sh [PORT]
# ============================================================

set -euo pipefail

# ----------------------
# Configuration
# ----------------------
PORT="${1:-1212}"
HOST="10.233.3.127"
API_BASE="http://${HOST}:${PORT}/v1"
API_KEY="any"
MODEL_VERSION="Qwen3.5-27B"
VIDEOSEEK_MODEL_VERSION="openai/${MODEL_VERSION}"

# Task Configuration
export HF_DATASETS_OFFLINE=1
export HF_HOME="/gemini/space/zyf"
export HF_DATASETS_CACHE="/gemini/space/gjx/lmms-eval/.cache/hf_datasets"
TASKS="${TASKS:-videomme}"
DATASET_PATH="/gemini/space/zyf/datasets/lmms-lab/Video-MME"

# Evaluation Configuration
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
LIMIT="${LIMIT:-}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/videoseek_qwen35_27b_videomme_api}"
LOG_SUFFIX="${LOG_SUFFIX:-videoseek_qwen35_27b_api_$(date +%Y%m%d_%H%M%S)}"
VERBOSITY="${VERBOSITY:-DEBUG}"

# VideoSeek args
MAX_STEPS="${MAX_STEPS:-6}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
TEMPERATURE="${TEMPERATURE:-0}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1800}"
SAMPLE_RETRY_ATTEMPTS="${SAMPLE_RETRY_ATTEMPTS:-0}"
SAMPLE_RETRY_BACKOFF_S="${SAMPLE_RETRY_BACKOFF_S:-2.0}"

# ----------------------
# Check Server Health
# ----------------------
echo "=============================================="
echo "VideoMME Evaluation via VideoSeek + vLLM API"
echo "=============================================="
echo ""
echo "Checking server health..."

if ! curl -s "${API_BASE}/../health" >/dev/null 2>&1; then
    echo "Warning: health endpoint is not available at ${API_BASE}/../health"
    echo "Continuing anyway because some OpenAI-compatible services do not expose /health."
else
    echo "Server is running at ${API_BASE}"
fi
echo ""

# ----------------------
# Print Configuration
# ----------------------
echo "Configuration:"
echo "  API Base URL:   ${API_BASE}"
echo "  Model:          ${VIDEOSEEK_MODEL_VERSION}"
echo "  Tasks:          ${TASKS}"
echo "  Batch Size:     ${BATCH_SIZE}"
echo "  Workers:        ${NUM_WORKERS}"
echo "  Dataset:        ${DATASET_PATH}"
echo "  Output Path:    ${OUTPUT_PATH}"
echo "  Limit:          ${LIMIT:-<none>}"
echo "  Max Steps:      ${MAX_STEPS}"
echo "  Max Tokens:     ${MAX_TOKENS}"
echo "  Retry Attempts: ${SAMPLE_RETRY_ATTEMPTS}"
echo ""
echo "=============================================="
echo ""

# ----------------------
# Run Evaluation
# ----------------------
python -m lmms_eval \
  --model videoseek \
  --force_simple \
  --model_args "model_name=${VIDEOSEEK_MODEL_VERSION},api_base=${API_BASE},api_key=${API_KEY},max_steps=${MAX_STEPS},max_tokens=${MAX_TOKENS},reasoning_effort=${REASONING_EFFORT},temperature=${TEMPERATURE},timeout=${REQUEST_TIMEOUT},num_workers=${NUM_WORKERS},sample_retry_attempts=${SAMPLE_RETRY_ATTEMPTS},sample_retry_backoff_s=${SAMPLE_RETRY_BACKOFF_S}" \
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
