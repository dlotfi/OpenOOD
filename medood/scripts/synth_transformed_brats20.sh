#!/bin/bash

source ./scripts/common_env.sh

python transform_brats20.py \
    --base_dir="$PROCESSED_DATASETS_DIR/brats20_t1/" \
    --output_dir="$PROCESSED_DATASETS_DIR/brats20_t1_transformed/" \
    --seed=$SEED \
    --skip_existing

python generate_imglist.py \
    --input_dir="$PROCESSED_DATASETS_DIR/brats20_t1_transformed" \
    --base_dir="$PROCESSED_DATASETS_DIR" \
    --output_dir="$IMGLIST_DIR"
