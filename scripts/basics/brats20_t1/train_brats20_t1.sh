#!/bin/bash
# sh scripts/basics/brats20_t1/train_brats20_t1.sh

SEED=0
MARK="default"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# train
python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/networks/resnet3d_18.yml \
    configs/pipelines/train/train_med3d.yml \
    configs/preprocessors/med3d_preprocessor.yml \
    --seed ${SEED} \
    --mark ${MARK}
