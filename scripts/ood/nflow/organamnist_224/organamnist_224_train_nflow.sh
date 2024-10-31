#!/bin/bash
# sh scripts/ood/nflow/organamnist_224/organamnist_224_train_nflow.sh

SEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# feature extraction
python main.py \
    --config configs/datasets/medmnist/organamnist_224.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/train/train_nflow_feat_extract.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.checkpoint "results/organamnist_resnet18_224x224/s${SEED}/resnet18_224_1.pth" \
    --network.checkpoint_key "net" \
    --seed ${SEED}

# train
python main.py \
    --config configs/datasets/medmnist/organamnist_224.yml \
    configs/networks/nflow_resnet18_224x224.yml \
    configs/pipelines/train/train_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/nflow.yml \
    --dataset.feat_root "./results/organamnist_resnet18_224x224_feat_extract_nflow_default/s${SEED}" \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint "./results/organamnist_resnet18_224x224/s${SEED}/resnet18_224_1.pth" \
    --network.backbone.checkpoint_key "net" \
    --seed ${SEED}
