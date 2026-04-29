#!/bin/bash

# ============================================================
# VideoMME Evaluation Script via OpenAI-Compatible API
# ============================================================
# Directly evaluates the API model with lmms-eval's openai backend.
# This does not call VideoSeek.
#
# Usage:
#   LIMIT="--limit 2" bash eval_videomme_direct_27b_api.sh
# ============================================================

set -euo pipefail


# ----------------------
# API Configuration
# ----------------------
API_URL="${API_URL:-http://10.233.3.127:1212/v1/chat/completions}"
API_BASE="${API_BASE:-${API_URL%/chat/completions}}"
API_KEY="${API_KEY:-any}"
MODEL_VERSION="${MODEL_NAME:-Qwen3.5-27B}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"

# ----------------------
# Environment
# ----------------------
REPO_ROOT="/gemini/space/gjx/lmms-eval"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-/gemini/space/zyf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/gemini/space/gjx/lmms-eval/.cache/hf_datasets}"

# ----------------------
# Task / Evaluation
# ----------------------
TASKS="${TASKS:-videomme_w_subtitle}"
BATCH_SIZE="${BATCH_SIZE:-6}"
LIMIT="${LIMIT:-}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/direct_qwen35_27b_videomme_w_subtitle_api}"
LOG_SUFFIX="${LOG_SUFFIX:-direct_qwen35_27b_api_$(date +%Y%m%d_%H%M%S)}"
VERBOSITY="${VERBOSITY:-INFO}"

# Generation / client args
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
NUM_CONCURRENT="${NUM_CONCURRENT:-6}"
MAX_RETRIES="${MAX_RETRIES:-3}"

echo "=============================================="
echo "VideoMME Evaluation via Direct API"
echo "=============================================="
echo "  API URL:      ${API_URL}"
echo "  API Base:     ${API_BASE}"
echo "  Model:        ${MODEL_VERSION}"
echo "  Python:       ${PYTHON_BIN}"
echo "  Tasks:        ${TASKS}"
echo "  Batch Size:   ${BATCH_SIZE}"
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
  --model openai \
  --force_simple \
  --model_args "model_version=${MODEL_VERSION},base_url=${API_BASE},api_key=${API_KEY},timeout=${REQUEST_TIMEOUT},num_concurrent=${NUM_CONCURRENT},max_retries=${MAX_RETRIES}" \
  --gen_kwargs "max_new_tokens=${MAX_NEW_TOKENS}" \
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
