#!/bin/bash

# ============================================================
# vLLM OpenAI-Compatible API Server Startup Script
# ============================================================
# This script starts a vLLM server for Qwen3.5-9B that provides
# an OpenAI-compatible API endpoint.
#
# Usage:
#   bash start_server.sh [PORT]
#
# Examples:
#   bash start_server.sh        # Default port 8000
#   bash start_server.sh 9000   # Custom port 9000
# ============================================================

# ----------------------
# Configuration
# ----------------------
MODEL_PATH="/gemini/space/zyf/models/Qwen/Qwen3___5-9B"
PORT="${1:-8000}"
HOST="0.0.0.0"

# GPU Configuration
TENSOR_PARALLEL_SIZE=2
DATA_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.9

# Model Configuration
MAX_MODEL_LEN=20000

# ----------------------
# Environment Setup
# ----------------------
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

# Check if venv exists
if [ ! -d "${VENV_DIR}" ]; then
    echo "Error: Virtual environment not found at ${VENV_DIR}"
    echo "Please create it first:"
    echo "  python -m venv ${VENV_DIR}"
    echo "  ${VENV_DIR}/bin/pip install vllm"
    exit 1
fi

# Activate venv
source "${VENV_DIR}/bin/activate"

# ----------------------
# Check if port is already in use
# ----------------------
# Using 'ss' as it is more likely to be installed by default than 'lsof'
if ss -tuln | grep -q ":${PORT} "; then
    echo "Error: Port ${PORT} is already in use!"
    echo "Please stop the existing server or use a different port."
    exit 1
fi

# ----------------------
# Start vLLM Server
# ----------------------
echo "Starting vLLM server on ${HOST}:${PORT}..."

/gemini/space/zyf/lmms-eval/evaluation_scripts/vllm_qwen35/.venv/bin/vllm serve "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --data-parallel-size "${DATA_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code \
    --served-model-name "qwen3.5-9b" \
    --enforce-eager \
    2>&1 | tee "vllm.log"