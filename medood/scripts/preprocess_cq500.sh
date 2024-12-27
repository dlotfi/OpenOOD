#!/bin/bash

source ./scripts/common_env.sh

python preprocess_cq500.py \
    --base_dir="$RAW_DATASETS_DIR/CQ500/images/" \
    --output_dir="$PROCESSED_DATASETS_DIR/cq500_ct/" \
    --num_samples=150 \
    --seed=$SEED \
    --use_gpu \
    --skip_existing

python generate_imglist.py \
    --input_dir="$PROCESSED_DATASETS_DIR/cq500_ct/" \
    --base_dir="$PROCESSED_DATASETS_DIR" \
    --output_dir="$IMGLIST_DIR"
