#!/bin/bash

# ============================================================
# VideoMME Evaluation Script via VideoSeek + OpenAI API
# ============================================================
# Uses lmms-eval's videoseek wrapper. VideoSeek calls an
# OpenAI-compatible inference service through LiteLLM.
#
# Usage:
#   LIMIT="--limit 2" bash eval_videomme_videoseek_397b_api.sh
# ============================================================

set -euo pipefail

# ----------------------
# API Configuration
# ----------------------
API_URL="${API_URL:-http://10.233.43.145:5590/v1/chat/completions}"
API_BASE="${API_BASE:-${API_URL%/chat/completions}}"
API_KEY="${API_KEY:-any}"
RAW_MODEL_NAME="${MODEL_NAME:-Qwen3.5-397B-A17B-FP8}"
MODEL_NAME_FOR_LITELLM="${MODEL_NAME_FOR_LITELLM:-openai/${RAW_MODEL_NAME}}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"

# ----------------------
# Environment
# ----------------------
REPO_ROOT="/gemini/space/gjx/lmms-eval"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-/gemini/space/zyf}"

# ----------------------
# Task / Evaluation
# ----------------------
TASKS="${TASKS:-videomme_w_subtitle}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LIMIT="${LIMIT:-}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/videoseek_qwen35_397b_videomme_w_subtitle_api}"
LOG_SUFFIX="${LOG_SUFFIX:-videoseek_qwen35_397b_api_$(date +%Y%m%d_%H%M%S)}"
VERBOSITY="${VERBOSITY:-INFO}"

# VideoSeek args
MAX_STEPS="${MAX_STEPS:-6}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
TEMPERATURE="${TEMPERATURE:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"
SAMPLE_RETRY_ATTEMPTS="${SAMPLE_RETRY_ATTEMPTS:-0}"
SAMPLE_RETRY_BACKOFF_S="${SAMPLE_RETRY_BACKOFF_S:-2.0}"

echo "=============================================="
echo "VideoMME Evaluation via VideoSeek"
echo "=============================================="
echo "  API URL:      ${API_URL}"
echo "  API Base:     ${API_BASE}"
echo "  Model:        ${MODEL_NAME_FOR_LITELLM}"
echo "  Python:       ${PYTHON_BIN}"
echo "  Tasks:        ${TASKS}"
echo "  Batch Size:   ${BATCH_SIZE}"
echo "  Workers:      ${NUM_WORKERS}"
echo "  Limit:        ${LIMIT:-<none>}"
echo "  Output Path:  ${OUTPUT_PATH}"
echo "=============================================="
echo ""

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Error: Python not found or not executable: ${PYTHON_BIN}"
    exit 1
fi

if ! curl -s --max-time 5 "${API_BASE}/models" >/dev/null 2>&1; then
    echo "Warning: Could not verify ${API_BASE}/models. Continuing anyway."
fi

cd "${REPO_ROOT}"

"${PYTHON_BIN}" -m lmms_eval \
  --model videoseek \
  --model_args "model_name=${MODEL_NAME_FOR_LITELLM},api_base=${API_BASE},api_key=${API_KEY},max_steps=${MAX_STEPS},max_tokens=${MAX_TOKENS},reasoning_effort=${REASONING_EFFORT},temperature=${TEMPERATURE},timeout=${REQUEST_TIMEOUT},num_workers=${NUM_WORKERS},sample_retry_attempts=${SAMPLE_RETRY_ATTEMPTS},sample_retry_backoff_s=${SAMPLE_RETRY_BACKOFF_S}" \
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
