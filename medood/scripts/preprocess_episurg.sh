#!/bin/bash

python preprocess_episurg.py \
    --base_dir="$RAW_DATASETS_DIR/EPISURG/EPISURG/subjects/" \
    --output_dir="$PROCESSED_DATASETS_DIR/episurg_t1/" \
    --num_samples=250 \
    --seed=328131023 \
    --use_gpu
