#!/bin/bash

source ./scripts/common_env.sh

python preprocess_chaos.py \
    --base_dir="$RAW_DATASETS_DIR/CHAOS/" \
    --output_dir="$PROCESSED_DATASETS_DIR/chaos_t1/" \
    --num_samples=80 \
    --seed=$SEED

python generate_imglist.py \
    --input_dir="$PROCESSED_DATASETS_DIR/chaos_t1/" \
    --base_dir="$PROCESSED_DATASETS_DIR" \
    --output_dir="$IMGLIST_DIR"
