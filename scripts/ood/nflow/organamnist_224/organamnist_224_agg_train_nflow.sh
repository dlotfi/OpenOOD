#!/bin/bash
# sh scripts/ood/nflow/organamnist_224/organamnist_224_agg_train_nflow.sh

SEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# feature extraction
python main.py \
    --config configs/datasets/medmnist/organamnist_224.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/networks/resnet18_224x224_feat_concat.yml \
    configs/pipelines/train/train_nflow_feat_extract.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained False \
    --network.encoder.pretrained True \
    --network.encoder.checkpoint "results/organamnist_resnet18_224x224/s${SEED}/resnet18_224_1.pth" \
    --network.encoder.checkpoint_key "net" \
    --seed ${SEED}

# train
python main.py \
    --config configs/datasets/medmnist/organamnist_224.yml \
    configs/networks/nflow_resnet18_224x224_feat_agg.yml \
    configs/pipelines/train/train_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/nflow.yml \
    --dataset.feat_root "./results/organamnist_feat_concat_feat_extract_nflow_default/s${SEED}" \
    --network.backbone.encoder.pretrained True \
    --network.backbone.encoder.checkpoint "./results/organamnist_resnet18_224x224/s${SEED}/resnet18_224_1.pth" \
    --network.backbone.encoder.checkpoint_key "net" \
    --seed ${SEED}
    # --optimizer.weight_decay 0.0001  # in case of non-finite gradient norm error
