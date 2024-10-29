#!/bin/bash

source ./scripts/common_env.sh

python preprocess_brats20.py \
    --base_dir="$RAW_DATASETS_DIR/BraTS_2020/MICCAI_BraTS2020_TrainingData/" \
    --output_dir="$PROCESSED_DATASETS_DIR/brats20_t1/" \
    --output_dir_t1c="$PROCESSED_DATASETS_DIR/brats20_t1c/" \
    --output_dir_t2f="$PROCESSED_DATASETS_DIR/brats20_t2f/" \
    --split_num_samples 274 20 75 \
    --seed=$SEED

python generate_imglist.py \
    --input_dir="$PROCESSED_DATASETS_DIR/brats20_t1/" \
    --base_dir="$PROCESSED_DATASETS_DIR" \
    --output_dir="$IMGLIST_DIR" \
    --labels LGG HGG

python generate_imglist.py \
    --input_dir="$PROCESSED_DATASETS_DIR/brats20_t1c/" \
    --base_dir="$PROCESSED_DATASETS_DIR" \
    --output_dir="$IMGLIST_DIR" \
    --labels LGG HGG

python generate_imglist.py \
    --input_dir="$PROCESSED_DATASETS_DIR/brats20_t2f/" \
    --base_dir="$PROCESSED_DATASETS_DIR" \
    --output_dir="$IMGLIST_DIR" \
    --labels LGG HGG
