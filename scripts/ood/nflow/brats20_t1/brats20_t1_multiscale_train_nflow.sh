#!/bin/bash
# sh scripts/ood/nflow/brats20_t1/brats20_t1_multiscale_train_nflow.sh

SEED=0
MARK1="5_feats"
MARK2="symmetric"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# feature extraction
python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    configs/networks/resnet3d_18_feat_concat.yml \
    configs/pipelines/train/train_nflow_feat_extract.yml \
    configs/preprocessors/med3d_preprocessor.yml \
    --network.pretrained False \
    --network.encoder.pretrained True \
    --network.encoder.checkpoint "./results/brats20_t1_resnet3d_18_med3d_e100_lr0.0001_default/s${SEED}/best.ckpt" \
    --seed ${SEED} \
    --mark ${MARK1}

# train
python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/networks/nflow_resnet3d_18_multiscale.yml \
    configs/pipelines/train/train_nflow.yml \
    configs/postprocessors/nflow.yml \
    --dataset.feat_root "./results/brats20_t1_feat_concat_feat_extract_nflow_${MARK1}/s${SEED}" \
    --seed ${SEED} \
    --mark "${MARK1}_${MARK2}"
