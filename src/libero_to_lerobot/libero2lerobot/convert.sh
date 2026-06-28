#!/bin/bash
export SVT_LOG=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=TRUE
export HDF5_USE_FILE_LOCKING=FALSE

SUITES=("libero_goal" "libero_spatial" "libero_object")

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

for suite in "${SUITES[@]}"; do
    echo "Converting suite: $suite"
    python libero_h5.py \
        --src-paths "$BASE_DIR/data/LIBERO-datasets-augmented/$suite" \
        --output-path "$BASE_DIR/data/lerobot_format/${suite}_augmented" \
        --executor local \
        --tasks-per-job 3 \
        --workers 10
done