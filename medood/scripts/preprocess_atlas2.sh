#!/bin/bash

source ./scripts/common_env.sh

python preprocess_atlas2.py \
    --base_dir="$RAW_DATASETS_DIR/ATLAS_2/ATLAS_2/" \
    --output_dir="$PROCESSED_DATASETS_DIR/atlas2_t1/" \
    --num_samples=150 \
    --seed=$SEED \
    --use_gpu \
    --skip_existing

python generate_imglist.py \
    --input_dir="$PROCESSED_DATASETS_DIR/atlas2_t1/" \
    --base_dir="$PROCESSED_DATASETS_DIR" \
    --output_dir="$IMGLIST_DIR"
