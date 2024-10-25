#!/bin/bash

python preprocess_cq500.py \
    --base_dir="$RAW_DATASETS_DIR/CQ500/images/" \
    --output_dir="$PROCESSED_DATASETS_DIR/cq500_ct/" \
    --num_samples=250 \
    --seed=328131023 \
    --use_gpu
