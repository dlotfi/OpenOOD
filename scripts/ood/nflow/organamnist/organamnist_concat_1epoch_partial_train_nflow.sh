#!/bin/bash
# sh scripts/ood/nflow/organamnist/organamnist_concat_1epoch_partial_train_nflow.sh

SEED=0
MARK="5_feats_10p"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# feature extraction
python main.py \
    --config configs/datasets/medmnist/organamnist_10p.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/networks/resnet18_28x28_feat_concat.yml \
    configs/pipelines/train/feat_extract_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained False \
    --network.encoder.pretrained True \
    --network.encoder.checkpoint "results/organamnist_resnet18_28x28/s${SEED}/resnet18_28_1.pth" \
    --network.encoder.checkpoint_key "net" \
    --seed ${SEED} \
    --mark ${MARK}

# train
python main.py \
    --config configs/datasets/medmnist/organamnist_10p.yml \
    configs/networks/nflow_resnet18_28x28_feat_concat.yml \
    configs/pipelines/train/train_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/nflow.yml \
    --dataset.feat_root "./results/organamnist_feat_concat_feat_extract_nflow_${MARK}/s${SEED}" \
    --optimizer.num_epochs 1 \
    --seed ${SEED} \
    --mark ${MARK}
