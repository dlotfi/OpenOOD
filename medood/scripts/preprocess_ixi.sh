#!/bin/bash

python preprocess_ixi.py \
    --base_dir="$RAW_DATASETS_DIR/IXI/images" \
    --output_dir="$PROCESSED_DATASETS_DIR/ixi_t1/" \
    --num_samples=250 \
    --seed=328131023 \
    --use_gpu
