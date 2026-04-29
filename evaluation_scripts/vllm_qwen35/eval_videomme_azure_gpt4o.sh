#!/bin/bash

# ============================================================
# VideoMME Evaluation Script via Azure OpenAI GPT-4o
# ============================================================

set -euo pipefail

ROOT_DIR="/gemini/space/gjx/lmms-eval"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/.env}"

if [ ! -f "${ENV_FILE}" ]; then
    echo "Error: .env file not found: ${ENV_FILE}"
    exit 1
fi

set -a
source "${ENV_FILE}"
set +a

: "${AZURE_OPENAI_API_KEY:?AZURE_OPENAI_API_KEY is required}"
: "${AZURE_OPENAI_DEPLOYMENT_NAME:?AZURE_OPENAI_DEPLOYMENT_NAME is required}"
: "${AZURE_OPENAI_API_VERSION:?AZURE_OPENAI_API_VERSION is required}"
: "${AZURE_OPENAI_API_BASE:=${AZURE_OPENAI_ENDPOINT:?AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_BASE is required}}"
export AZURE_OPENAI_API_KEY AZURE_OPENAI_API_VERSION AZURE_OPENAI_API_BASE

export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-/gemini/space/zyf}"

TASKS="${TASKS:-videomme}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_CONCURRENT="${NUM_CONCURRENT:-4}"
LIMIT="${LIMIT:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-384}"
TIMEOUT="${TIMEOUT:-180}"
MAX_RETRIES="${MAX_RETRIES:-3}"
VERBOSITY="${VERBOSITY:-INFO}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/azure_gpt4o_videomme}"
LOG_SUFFIX="azure_gpt4o_$(date +%Y%m%d_%H%M%S)"

cd "${ROOT_DIR}"

echo "=============================================="
echo "VideoMME Evaluation via Azure GPT-4o"
echo "=============================================="
echo "  Endpoint:       ${AZURE_OPENAI_API_BASE}"
echo "  Deployment:     ${AZURE_OPENAI_DEPLOYMENT_NAME}"
echo "  API Version:    ${AZURE_OPENAI_API_VERSION}"
echo "  Tasks:          ${TASKS}"
echo "  Batch Size:     ${BATCH_SIZE}"
echo "  Concurrent:     ${NUM_CONCURRENT}"
echo "  Max Frames:     ${MAX_FRAMES_NUM}"
echo "  Max New Tokens: ${MAX_NEW_TOKENS}"
echo "  Limit:          ${LIMIT:-<none>}"
echo "  Output Path:    ${OUTPUT_PATH}"
echo "=============================================="

python -m lmms_eval \
  --model openai \
  --force_simple \
  --model_args "model_version=${AZURE_OPENAI_DEPLOYMENT_NAME},azure_openai=True,httpx_trust_env=False,timeout=${TIMEOUT},max_retries=${MAX_RETRIES},max_frames_num=${MAX_FRAMES_NUM},num_concurrent=${NUM_CONCURRENT}" \
  --gen_kwargs "max_new_tokens=${MAX_NEW_TOKENS},temperature=0" \
  --tasks "${TASKS}" \
  --batch_size "${BATCH_SIZE}" \
  ${LIMIT} \
  --log_samples \
  --log_samples_suffix "${LOG_SUFFIX}" \
  --output_path "${OUTPUT_PATH}" \
  --verbosity "${VERBOSITY}"

echo "=============================================="
echo "Evaluation completed. Results saved to: ${OUTPUT_PATH}"
echo "=============================================="
