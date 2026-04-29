#!/bin/bash

# ============================================================
# VideoMME Dataset Preparation Script
# ============================================================
# This script extracts video files from the downloaded zip
# archives to the correct directory structure for lmms-eval.
#
# Usage:
#   bash prepare_videomme_dataset.sh
# ============================================================

# ----------------------
# Configuration
# ----------------------
DATASET_ROOT="/gemini/space/zyf/datasets/lmms-lab/Video-MME"
TARGET_DIR="${DATASET_ROOT}/videomme/data"
SUBTITLE_DIR="${DATASET_ROOT}/videomme/subtitle"

# ----------------------
# Create Directories
# ----------------------
echo "Creating directory structure..."
mkdir -p "${TARGET_DIR}"
mkdir -p "${SUBTITLE_DIR}"

# ----------------------
# Extract Videos
# ----------------------
echo ""
echo "Extracting video files..."

cd "${DATASET_ROOT}"

# Extract all video zip files
for zip_file in videos_chunked_*.zip; do
    if [ -f "$zip_file" ]; then
        echo "Extracting ${zip_file}..."
        unzip -o -q "$zip_file" -d "${TARGET_DIR}/"
    fi
done

# ----------------------
# Extract Subtitles
# ----------------------
echo ""
echo "Extracting subtitle files..."

if [ -f "subtitle.zip" ]; then
    unzip -o -q "subtitle.zip" -d "${SUBTITLE_DIR}/"
    echo "Subtitles extracted."
else
    echo "Warning: subtitle.zip not found"
fi

# ----------------------
# Verify Extraction
# ----------------------
echo ""
echo "=============================================="
echo "Dataset Preparation Complete!"
echo "=============================================="
echo ""
echo "Video directory: ${TARGET_DIR}"
echo "Subtitle directory: ${SUBTITLE_DIR}"
echo ""
echo "Video files count: $(ls -1 "${TARGET_DIR}"/*.mp4 2>/dev/null | wc -l)"
echo "Subtitle files count: $(ls -1 "${SUBTITLE_DIR}"/*.srt 2>/dev/null | wc -l)"
echo ""
echo "To use this dataset, set:"
echo "  export HF_HOME=\"${DATASET_ROOT}\""
echo "=============================================="
