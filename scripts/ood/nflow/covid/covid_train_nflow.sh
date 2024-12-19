#!/bin/bash
# sh scripts/ood/nflow/covid/covid_train_nflow.sh

SEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# feature extraction
python main.py \
    --config configs/datasets/covid/covid.yml \
    configs/datasets/covid/covid_ood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/train/train_nflow_feat_extract.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.checkpoint "./results/covid_resnet18_224x224_base_e200_lr0.0001_default/s${SEED}/best.ckpt" \
    --seed ${SEED}

# train
python main.py \
    --config configs/datasets/covid/covid.yml \
    configs/networks/nflow_resnet18_224x224.yml \
    configs/pipelines/train/train_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/nflow.yml \
    --dataset.feat_root "./results/covid_resnet18_224x224_feat_extract_nflow_default/s${SEED}" \
    --network.nflow.l2_normalize True \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint "./results/covid_resnet18_224x224_base_e200_lr0.0001_default/s${SEED}/best.ckpt" \
    --seed ${SEED}
