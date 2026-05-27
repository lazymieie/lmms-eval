
set -euo pipefail

# ----------------------
# Configuration
# ----------------------

# API_BASE="http://10.233.114.36:8000/v1|http://10.233.92.52:8000/v1|http://127.0.0.1:8000/v1|http://10.233.48.11:5590/v1"
API_BASE="http://10.233.48.11:5590/v1"
API_KEY="${API_KEY:-any}"
MODEL_VERSION="${MODEL_VERSION:-Qwen3.5-27B}"

# Task Configuration
export HF_DATASETS_OFFLINE=1
export HF_HOME="${HF_HOME:-/gemini/space/zyf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/gemini/space/gjx/lmms-eval/.cache/hf_datasets}"
TASKS="${TASKS:-videomme_long}"
DATASET_PATH="${DATASET_PATH:-/gemini/space/zyf/datasets/lmms-lab/Video-MME}"

# Evaluation Configuration
BATCH_SIZE="${BATCH_SIZE:-8}"
# LIMIT="${LIMIT:---limit 8}"
LIMIT="${LIMIT:-}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/qwen35_27b_videomme_long_wo_subtitle_api_frames165_sglang}"
LOG_SUFFIX="${LOG_SUFFIX:-qwen35_27b_videomme_long_wo_subtitle_api_frames165_$(date +%Y%m%d_%H%M%S)}"
VERBOSITY="${VERBOSITY:-DEBUG}"

# Model / Sampling Args
NFRAMES="${NFRAMES:-165}"
NUM_CPUS="${NUM_CPUS:-8}"
TIMEOUT="${TIMEOUT:-1800}"
MAX_RETRIES="${MAX_RETRIES:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128536}"
TEMPERATURE="${TEMPERATURE:-0}"

# ----------------------
# Check Server Health
# ----------------------
echo "=============================================="
echo "VideoMME-Long Evaluation via vLLM API"
echo "=============================================="
echo ""
echo "Checking server health..."

IFS='|' read -r -a API_BASES <<< "${API_BASE}"
for base in "${API_BASES[@]}"; do
  health_url="${base%/v1}/health"
  if ! curl -s "$health_url" >/dev/null 2>&1; then
    echo "Error: vLLM server is not running at $base"
    exit 1
  fi
done


echo "Server is running at ${API_BASE}"
echo ""

# ----------------------
# Print Configuration
# ----------------------
echo "Configuration:"
echo "  API Base URL:  ${API_BASE}"
echo "  Model:         ${MODEL_VERSION}"
echo "  Tasks:         ${TASKS}"
echo "  Batch Size:    ${BATCH_SIZE}"
echo "  NFrames:       ${NFRAMES}"
echo "  Num CPUs:      ${NUM_CPUS}"
echo "  Dataset:       ${DATASET_PATH}"
echo "  Output Path:   ${OUTPUT_PATH}"
echo "  Limit:         ${LIMIT:-<none>}"
echo ""
echo "=============================================="
echo ""

# ----------------------
# Run Evaluation
# ----------------------
python -m lmms_eval \
  --model async_openai \
  --model_args "model_version=${MODEL_VERSION},base_url=${API_BASE},api_key=${API_KEY},nframes=${NFRAMES},num_cpus=${NUM_CPUS},timeout=${TIMEOUT},max_retries=${MAX_RETRIES}" \
  --gen_kwargs "max_new_tokens=${MAX_NEW_TOKENS},temperature=${TEMPERATURE}" \
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
