#!/bin/bash
# sh scripts/osr/nflow/organamnist_concat_train_nflow.sh

SEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# feature extraction
python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/networks/resnet18_28x28_feat_concat.yml \
    configs/pipelines/train/train_nflow_feat_extract.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained False \
    --network.encoder.pretrained True \
    --network.encoder.checkpoint "results/organamnist_resnet18_28x28/s${SEED}/resnet18_28_1.pth" \
    --network.encoder.checkpoint_key "net" \
    --seed ${SEED}

# train
python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/networks/nflow_resnet18_28x28_feat_concat.yml \
    configs/pipelines/train/train_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/nflow.yml \
    --dataset.feat_root "./results/organamnist_feat_concat_feat_extract_nflow_default/s${SEED}" \
    --network.backbone.encoder.pretrained True \
    --network.backbone.encoder.checkpoint "./results/organamnist_resnet18_28x28/s${SEED}/resnet18_28_1.pth" \
    --network.backbone.encoder.checkpoint_key "net" \
    --seed ${SEED}
