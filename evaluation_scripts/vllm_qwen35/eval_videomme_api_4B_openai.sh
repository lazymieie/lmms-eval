#!/bin/bash

# ============================================================
# VideoMME Evaluation Script via vLLM API
# ============================================================
# This script evaluates Qwen3.5-9B on VideoMME using the
# vLLM OpenAI-compatible API server.
#
# Prerequisites:
#   1. Start the vLLM server: bash start_server.sh
#   2. Wait for the server to be ready
#   3. Run this script
#
# Usage:
#   bash eval_via_api.sh [PORT]
#
# Examples:
#   bash eval_via_api.sh        # Default port 8000
#   bash eval_via_api.sh 9000   # Custom port 9000
# ============================================================

set -euo pipefail

# ----------------------
# Configuration
# ----------------------
# vllm
PORT="${1:-5590}"
HOST="10.233.58.81"
API_BASE="http://${HOST}:${PORT}/v1"
API_KEY="any"  # vLLM doesn't require real API key, but needs a non-empty string
MODEL_VERSION="Qwen3.5-4B"

# Task Configuration
export HF_DATASETS_OFFLINE=1
export HF_HOME="/gemini/space/zyf"
export HF_DATASETS_CACHE="/gemini/space/gjx/lmms-eval/.cache/hf_datasets"
TASKS="videomme_long"
DATASET_PATH="/gemini/space/zyf/datasets/lmms-lab/Video-MME"

# Evaluation Configuration
BATCH_SIZE=10
# LIMIT="${LIMIT:---limit 10}"  # Set LIMIT="--limit 10" for testing
LIMIT="${LIMIT:-}"
OUTPUT_PATH="./logs/qwen35_4b_videomme_api_openai_long"
LOG_SUFFIX="qwen35_4b_api_openai_$(date +%Y%m%d_%H%M%S)"
VERBOSITY="${VERBOSITY:-DEBUG}"

#GEN ARGS
MAX_NEW_TOKENS=10240

# ----------------------
# Check Server Health
# ----------------------
echo "=============================================="
echo "VideoMME Evaluation via vLLM API"
echo "=============================================="
echo ""
echo "Checking server health..."

if ! curl -s "${API_BASE}/../health" >/dev/null 2>&1; then
    echo "Error: vLLM server is not running at ${API_BASE}"
    echo "Please start the server first:"
    echo "  bash evaluation_scripts/vllm/start_server.sh ${PORT}"
    exit 1
fi

echo "Server is running at ${API_BASE}"
echo ""

# ----------------------
# Print Configuration
# ----------------------
echo "Configuration:"
echo "  API Base URL:  ${API_BASE}"
echo "  Model:        ${MODEL_VERSION}"
echo "  Tasks:        ${TASKS}"
echo "  Batch Size:   ${BATCH_SIZE}"
echo "  Dataset:      ${DATASET_PATH}"
echo "  Output Path:  ${OUTPUT_PATH}"
echo ""
echo "=============================================="
echo ""

# ----------------------
# Run Evaluation
# ----------------------
# source /gemini/space/zyf/lmms-eval/.newvenv/bin/activate
python -m lmms_eval \
  --model openai \
  --force_simple \
  --model_args "model_version=${MODEL_VERSION},base_url=${API_BASE},api_key=${API_KEY}" \
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
