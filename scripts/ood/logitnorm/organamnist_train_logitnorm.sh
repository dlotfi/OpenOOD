#!/bin/bash
# sh scripts/ood/logitnorm/organamnist_train_logitnorm.sh

python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/networks/resnet18_28x28.yml \
    configs/pipelines/train/train_logitnorm.yml \
    configs/preprocessors/base_preprocessor.yml \
    --seed 0
