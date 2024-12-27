#!/bin/bash

source ./scripts/common_env.sh

python preprocess_wmh2017.py \
    --base_dir="$RAW_DATASETS_DIR/WMH_2017/" \
    --output_dir="$PROCESSED_DATASETS_DIR/wmh2017_t1/" \
    --num_samples=150 \
    --seed=$SEED \
    --use_gpu \
    --skip_existing

python generate_imglist.py \
    --input_dir="$PROCESSED_DATASETS_DIR/wmh2017_t1/" \
    --base_dir="$PROCESSED_DATASETS_DIR" \
    --output_dir="$IMGLIST_DIR"
