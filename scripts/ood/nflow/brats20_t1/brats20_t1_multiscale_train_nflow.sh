#!/bin/bash
# sh scripts/ood/nflow/brats20_t1/brats20_t1_multiscale_train_nflow.sh

SEED=0
MARK="default"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# feature extraction
python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    configs/networks/resnet3d_18_feat_concat.yml \
    configs/pipelines/train/train_nflow_feat_extract.yml \
    configs/preprocessors/med3d_preprocessor.yml \
    --network.pretrained False \
    --network.feat_agg.layers 1 2 3 4 5 \
    --network.encoder.pretrained True \
    --network.encoder.checkpoint "./results/brats20_t1_resnet3d_18_med3d_e100_lr0.0001_${MARK}/s${SEED}/best.ckpt" \
    --seed ${SEED} \
    --mark "${MARK}"

# train
python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/networks/nflow_resnet3d_18_multiscale.yml \
    configs/pipelines/train/train_nflow.yml \
    configs/postprocessors/nflow.yml \
    --dataset.feat_root "./results/brats20_t1_feat_concat_feat_extract_nflow_${MARK}/s${SEED}" \
    --network.nflow.normalize_input True \
    --network.nflow.latent_size 64 64 128 256 512 \
    --network.nflow.hidden_size 64 64 128 256 512 \
    --network.nflow.n_flows 4 4 4 4 4 \
    --network.backbone.encoder.pretrained True \
    --network.backbone.encoder.checkpoint "./results/brats20_t1_resnet3d_18_med3d_e100_lr0.0001_${MARK}/s${SEED}/best.ckpt" \
    --optimizer.grad_clip_norm 1.0 \
    --optimizer.weight_decay 0.0001 \
    --recorder.save_last_model True \
    --seed ${SEED} \
    --mark "${MARK}_multiscale"
