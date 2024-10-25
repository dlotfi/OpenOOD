#!/bin/bash

python preprocessing/transform_brats20.py \
    --base_dir="$PROCESSED_DATASETS_DIR/brats20_t1/" \
    --output_dir="$PROCESSED_DATASETS_DIR/brats20_t1_transformed/" \
    --seed=328131023
