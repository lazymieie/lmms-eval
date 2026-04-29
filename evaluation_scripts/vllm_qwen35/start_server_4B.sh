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
MODEL_PATH="/gemini/space/gjx/models/Qwen/Qwen3.5-4B"
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
# Use the currently activated conda environment.
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "Error: no conda environment is active."
    echo "Please run: conda activate <env-name>"
    exit 1
fi

VLLM_BIN="${VLLM_BIN:-${CONDA_PREFIX}/bin/vllm}"

if ! command -v "${VLLM_BIN}" >/dev/null 2>&1; then
    echo "Error: vLLM executable not found: ${VLLM_BIN}"
    echo "Please install vLLM in the active conda environment:"
    echo "  pip install vllm"
    exit 1
fi

echo "Using conda:  ${CONDA_PREFIX}"
echo "Using Python: $(command -v python)"
echo "Using vLLM:   ${VLLM_BIN}"

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

"${VLLM_BIN}" serve "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --data-parallel-size "${DATA_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code \
    --served-model-name "qwen3.5-4b" \
    --enforce-eager \
    2>&1 | tee "vllm.log"
