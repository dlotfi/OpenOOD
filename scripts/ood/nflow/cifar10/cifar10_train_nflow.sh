#!/bin/bash
# sh scripts/ood/nflow/cifar10/cifar10_train_nflow.sh

SEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# feature extraction
python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/train_nflow_feat_extract.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.checkpoint "./results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s${SEED}/best.ckpt" \
    --seed ${SEED}

# train
python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/networks/nflow_resnet18_32x32.yml \
    configs/pipelines/train/train_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/nflow.yml \
    --dataset.feat_root "./results/cifar10_resnet18_32x32_feat_extract_nflow_default/s${SEED}" \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint "./results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s${SEED}/best.ckpt" \
    --seed ${SEED}
