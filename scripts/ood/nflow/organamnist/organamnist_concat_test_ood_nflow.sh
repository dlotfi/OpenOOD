#!/bin/bash
# sh scripts/ood/nflow/organamnist/organamnist_concat_test_ood_nflow.sh

SEED=0
MARK="5_feats"

python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/networks/nflow_resnet18_28x28_feat_concat.yml \
    configs/pipelines/test/feat_extract_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint "./results/organamnist_nflow_nflow_e100_lr0.0001_${MARK}/s${SEED}/best_nflow.ckpt" None \
    --network.backbone.pretrained False \
    --network.backbone.encoder.pretrained True \
    --network.backbone.encoder.checkpoint "./results/organamnist_resnet18_28x28/s${SEED}/resnet18_28_1.pth" \
    --network.backbone.encoder.checkpoint_key "net" \
    --seed ${SEED} \
    --mark ${MARK}

python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/networks/nflow_resnet18_28x28_feat_concat.yml \
    configs/pipelines/test/test_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/nflow.yml \
    --dataset.feat_root "./results/organamnist_feat_concat_feat_extract_nflow_${MARK}/s${SEED}" \
    --ood_dataset.feat_root "./results/organamnist_nflow_feat_extract_nflow_${MARK}/s${SEED}" \
    --network.pretrained True \
    --network.checkpoint "./results/organamnist_nflow_nflow_e100_lr0.0001_${MARK}/s${SEED}/best_nflow.ckpt" None \
    --network.backbone.pretrained False \
    --seed ${SEED} \
    --mark ${MARK}
