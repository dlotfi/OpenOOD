#!/bin/bash

python preprocessing/preprocess_brats23_ssa.py \
    --base_dir="$RAW_DATASETS_DIR/BraTS_2023/BraTS-SSA/" \
    --output_dir="$PROCESSED_DATASETS_DIR/brats23_ssa_t1/" \
    --num_samples=60 \
    --seed=328131023
