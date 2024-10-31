#!/bin/bash
# sh scripts/ood/nflow/organamnist_224/organamnist_224_concat_test_ood_nflow.sh

SEED=0
python main.py \
    --config configs/datasets/medmnist/organamnist_224.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/networks/nflow_resnet18_224x224_feat_concat.yml \
    configs/pipelines/test/test_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/nflow.yml \
    --num_workers 8 \
    --network.pretrained True \
    --network.checkpoint "./results/organamnist_nflow_nflow_e100_lr0.0001_default/s${SEED}/best_nflow.ckpt" None \
    --network.backbone.pretrained False \
    --network.backbone.encoder.pretrained True \
    --network.backbone.encoder.checkpoint "./results/organamnist_resnet18_224x224/s${SEED}/resnet18_224_1.pth" \
    --network.backbone.encoder.checkpoint_key "net" \
    --seed ${SEED}
