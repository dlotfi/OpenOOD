#!/bin/bash

source ./scripts/common_env.sh

python preprocess_lumbar.py \
    --base_dir="$RAW_DATASETS_DIR/Lumbar_Spine/01_MRI_Data/" \
    --output_dir="$PROCESSED_DATASETS_DIR/lumbar_t1/" \
    --num_samples=150 \
    --seed=$SEED

python generate_imglist.py \
    --input_dir="$PROCESSED_DATASETS_DIR/lumbar_t1/" \
    --base_dir="$PROCESSED_DATASETS_DIR" \
    --output_dir="$IMGLIST_DIR"
