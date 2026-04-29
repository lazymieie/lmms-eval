#!/bin/bash

# ============================================================
# vLLM OpenAI-Compatible API Server Startup Script
# ============================================================
# Starts Qwen3.5-4B with 8 GPUs via tensor parallelism and
# data parallelism. This script uses the currently activated
# conda environment and does not use the local .venv.
#
# Usage:
#   bash start_server_4B_8gpu.sh [PORT]
#
# Examples:
#   bash start_server_4B_8gpu.sh        # Default port 8000
#   bash start_server_4B_8gpu.sh 9000   # Custom port 9000
# ============================================================

set -euo pipefail

# ----------------------
# Configuration
# ----------------------
MODEL_PATH="/gemini/space/gjx/models/Qwen/Qwen3.5-4B"
PORT="${1:-8000}"
HOST="0.0.0.0"

# GPU Configuration
# Total GPUs = TENSOR_PARALLEL_SIZE * DATA_PARALLEL_SIZE = 2 * 4 = 8.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TENSOR_PARALLEL_SIZE=2
DATA_PARALLEL_SIZE=4
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
echo "Using GPUs:   ${CUDA_VISIBLE_DEVICES}"
echo "TP x DP:      ${TENSOR_PARALLEL_SIZE} x ${DATA_PARALLEL_SIZE}"

# ----------------------
# Check if port is already in use
# ----------------------
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
    2>&1 | tee "vllm_4b_8gpu.log"
