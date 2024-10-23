#!/bin/bash

python preprocessing/preprocess_atlas2.py \
    --base_dir="$RAW_DATASETS_DIR/ATLAS_2/ATLAS_2/" \
    --output_dir="$PROCESSED_DATASETS_DIR/atlas2_t1/" \
    --num_samples=250 \
    --seed=328131023 \
    --use_gpu
