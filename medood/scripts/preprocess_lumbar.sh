#!/bin/bash

python preprocess_lumbar.py \
    --base_dir="$RAW_DATASETS_DIR/Lumbar_Spine_MRI/01_MRI_Data/" \
    --output_dir="$PROCESSED_DATASETS_DIR/lumbar_t1/" \
    --num_samples=250 \
    --seed=328131023
