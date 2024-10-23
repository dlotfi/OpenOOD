#!/bin/bash

python preprocessing/preprocess_wmh2017.py \
    --base_dir="$RAW_DATASETS_DIR/WMH_2017/" \
    --output_dir="$PROCESSED_DATASETS_DIR/wmh2017_t1/" \
    --num_samples=160 \
    --seed=328131023 \
    --use_gpu
