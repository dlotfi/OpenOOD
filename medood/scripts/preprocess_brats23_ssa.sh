#!/bin/bash

source ./scripts/common_env.sh

python preprocess_brats23_ssa.py \
    --base_dir="$RAW_DATASETS_DIR/BraTS_2023/BraTS-SSA/" \
    --output_dir="$PROCESSED_DATASETS_DIR/brats23_ssa_t1/" \
    --num_samples=60 \
    --seed=$SEED

python generate_imglist.py \
    --input_dir="$PROCESSED_DATASETS_DIR/brats23_ssa_t1/" \
    --base_dir="$PROCESSED_DATASETS_DIR" \
    --output_dir="$IMGLIST_DIR"
