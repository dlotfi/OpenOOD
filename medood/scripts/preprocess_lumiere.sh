#!/bin/bash

python preprocess_lumiere.py \
    --base_dir="$RAW_DATASETS_DIR/LUMIERE/Imaging" \
    --output_dir="$PROCESSED_DATASETS_DIR/lumiere_t1/" \
    --num_samples=80 \
    --seed=328131023 \
    --use_gpu

python generate_imglist.py \
    --input_dir="$PROCESSED_DATASETS_DIR/lumiere_t1/" \
    --base_dir="$PROCESSED_DATASETS_DIR" \
    --output_dir="$IMGLIST_DIR"
