#!/bin/bash

python preprocess_ixi.py \
    --base_dir="$RAW_DATASETS_DIR/IXI/images" \
    --output_dir="$PROCESSED_DATASETS_DIR/ixi_t1/" \
    --num_samples=250 \
    --seed=328131023 \
    --use_gpu

python generate_imglist.py \
    --input_dir="$PROCESSED_DATASETS_DIR/ixi_t1/" \
    --base_dir="$PROCESSED_DATASETS_DIR" \
    --output_dir="$IMGLIST_DIR"
