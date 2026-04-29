#!/bin/bash

set -euo pipefail

# Configuration
PORT="${1:-8000}"
HOST="10.233.86.120"
API_BASE="http://${HOST}:${PORT}/v1"
API_KEY="any"
EVAL_MODEL_VERSION="qwen3.5-9b"

# Task Configuration
TASKS="mmvu_val"
DATASET_PATH="/gemini/space/zyf"
export HF_HOME="${DATASET_PATH}"

# Evaluation Configuration
BATCH_SIZE=4
LIMIT="${LIMIT:-}"
OUTPUT_PATH="./logs/qwen35_9b_mmvu_api"
LOG_SUFFIX="qwen35_9b_mmvu_api_$(date +%Y%m%d_%H%M%S)"

# llm judge                                                                                                    
export OPENAI_API_URL="http://10.233.123.174:18000/v1"
export MODEL_VERSION="gpt-4o-2"
export OPENAI_API_KEY="any"

#GEN ARGS
MAX_NEW_TOKENS=10240

# Check Server Health
echo "=============================================="
echo "MMVU Evaluation via vLLM API"
echo "=============================================="
echo ""
echo "Checking server health..."

if ! curl -s "${API_BASE}/../health" >/dev/null 2>&1; then
    echo "Error: vLLM server is not running at ${API_BASE}"
    echo "Please start the server first"
    exit 1
fi

echo "Server is running at ${API_BASE}"
echo ""

# Print Configuration
echo "Configuration:"
echo "  API Base URL:  ${API_BASE}"
echo "  Model:        ${EVAL_MODEL_VERSION}"
echo "  Tasks:        ${TASKS}"
echo "  Batch Size:   ${BATCH_SIZE}"
echo "  Output Path:  ${OUTPUT_PATH}"
echo ""
echo "=============================================="
echo ""

# Run Evaluation
source /gemini/space/zyf/lmms-eval/.newvenv/bin/activate
python -m lmms_eval \
  --model openai \
  --model_args "model_version=${EVAL_MODEL_VERSION},base_url=${API_BASE},api_key=${API_KEY}" \
  --gen_kwargs "max_new_tokens=${MAX_NEW_TOKENS}" \
  --tasks "${TASKS}" \
  --batch_size "${BATCH_SIZE}" \
  ${LIMIT} \
  --log_samples \
  --log_samples_suffix "${LOG_SUFFIX}" \
  --output_path "${OUTPUT_PATH}"

echo ""
echo "=============================================="
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_PATH}"
echo "=============================================="