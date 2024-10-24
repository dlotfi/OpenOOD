#!/bin/bash

python preprocessing/preprocess_brats20.py \
    --base_dir="$RAW_DATASETS_DIR/BraTS_2020/MICCAI_BraTS2020_TrainingData/" \
    --output_dir="$PROCESSED_DATASETS_DIR/brats20_t1/" \
    --output_dir_t1c="$PROCESSED_DATASETS_DIR/brats20_t1c/" \
    --output_dir_t2f="$PROCESSED_DATASETS_DIR/brats20_t2f/" \
    --split_num_samples 276 18 75 \
    --seed=328131023
