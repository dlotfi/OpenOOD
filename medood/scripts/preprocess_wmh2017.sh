#!/bin/bash

python preprocess_wmh2017.py \
    --base_dir="$RAW_DATASETS_DIR/WMH_2017/" \
    --output_dir="$PROCESSED_DATASETS_DIR/wmh2017_t1/" \
    --num_samples=160 \
    --seed=328131023 \
    --use_gpu

python generate_imglist.py \
    --input_dir="$PROCESSED_DATASETS_DIR/wmh2017_t1/" \
    --base_dir="$PROCESSED_DATASETS_DIR" \
    --output_dir="$IMGLIST_DIR"
