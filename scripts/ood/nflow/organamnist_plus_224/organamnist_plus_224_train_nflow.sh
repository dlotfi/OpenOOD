#!/bin/bash
# sh scripts/ood/nflow/organamnist_plus_224/organamnist_plus_224_train_nflow.sh

SEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# feature extraction
python main.py \
    --config configs/datasets/medmnist_plus/organamnist_224.yml \
    configs/datasets/medmnist_plus/organamnist_224_ood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/train/train_nflow_feat_extract.yml \
    configs/preprocessors/base_preprocessor.yml \
    --dataset.normalization_type imagenet \
    --network.checkpoint "results/organamnist_224_resnet18_224/organamnist_224_endToEnd_resnet18_s9930641_best.pth" \
    --seed ${SEED}

# train
python main.py \
    --config configs/datasets/medmnist_plus/organamnist_224.yml \
    configs/networks/nflow_resnet18_224x224.yml \
    configs/pipelines/train/train_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/nflow.yml \
    --dataset.feat_root "./results/organamnist_224_resnet18_224x224_feat_extract_nflow_default/s${SEED}" \
    --network.nflow.l2_normalize True \
    --network.nflow.clamp_value 10.0 \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint "./results/organamnist_224_resnet18_224/organamnist_224_endToEnd_resnet18_s9930641_best.pth" \
    --optimizer.weight_decay 0.0001 \
    --seed ${SEED}
