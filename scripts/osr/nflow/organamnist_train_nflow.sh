#!/bin/bash
# sh scripts/osr/nflow/organamnist_train_nflow.sh

SEED=0

# feature extraction
python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/networks/resnet18_28x28.yml \
    configs/pipelines/train/train_nflow_feat_extract.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.checkpoint "results/organamnist_resnet18_28x28/s${SEED}/resnet18_28_1.pth" \
    --network.checkpoint_key "net" \
    --seed ${SEED}

# train
python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/networks/nflow_resnet18_28x28.yml \
    configs/pipelines/train/train_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/nflow.yml \
    --dataset.feat_root "./results/organamnist_resnet18_28x28_feat_extract_nflow_default/s${SEED}" \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint "./results/organamnist_resnet18_28x28/s${SEED}/resnet18_28_1.pth" \
    --network.backbone.checkpoint_key "net" \
    --seed ${SEED}
