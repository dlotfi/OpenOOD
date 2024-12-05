#!/bin/bash

source ./scripts/common_env.sh

python preprocess_brats20.py \
    --base_dir="$RAW_DATASETS_DIR/BraTS_2020/MICCAI_BraTS2020_TrainingData/" \
    --output_dir="$PROCESSED_DATASETS_DIR/brats20_t1/" \
    --extra_output_dirs "t2=$PROCESSED_DATASETS_DIR/brats20_t2/" "t1c=$PROCESSED_DATASETS_DIR/brats20_t1c/" "t2f=$PROCESSED_DATASETS_DIR/brats20_t2f/" \
    --split_num_samples 274 20 75 \
    --seed=$SEED

for modality in t1 t2 t1c t2f; do
  python generate_imglist.py \
      --input_dir="$PROCESSED_DATASETS_DIR/brats20_${modality}/" \
      --base_dir="$PROCESSED_DATASETS_DIR" \
      --output_dir="$IMGLIST_DIR" \
      --labels LGG HGG
done
