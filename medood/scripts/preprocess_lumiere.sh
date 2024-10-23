#!/bin/bash

python preprocessing/preprocess_lumiere.py \
    --base_dir="$RAW_DATASETS_DIR/LUMIERE/Imaging" \
    --output_dir="$PROCESSED_DATASETS_DIR/lumiere_t1/" \
    --num_samples=80 \
    --seed=328131023 \
    --use_gpu
