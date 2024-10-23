#!/bin/bash

python preprocessing/preprocess_chaos.py \
    --base_dir="$RAW_DATASETS_DIR/CHAOS/" \
    --output_dir="$PROCESSED_DATASETS_DIR/chaos_t1/" \
    --num_samples=80 \
    --seed=328131023
