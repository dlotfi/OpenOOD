#!/bin/bash
# sh scripts/ood/nflow/organamnist_plus_224/organamnist_plus_224_test_ood_nflow.sh

SEED=0
python main.py \
    --config configs/datasets/medmnist_plus/organamnist_224.yml \
    configs/datasets/medmnist_plus/organamnist_224_ood.yml \
    configs/networks/nflow_resnet18_224x224.yml \
    configs/pipelines/test/test_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/nflow.yml \
    --num_workers 8 \
    --dataset.normalization_type imagenet \
    --network.pretrained True \
    --network.checkpoint "./results/organamnist_224_nflow_nflow_e100_lr0.0001_default/s${SEED}/best_nflow.ckpt" None \
    --network.nflow.l2_normalize True \
    --network.nflow.clamp_value 10.0 \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint "results/organamnist_224_resnet18_224/organamnist_224_endToEnd_resnet18_s9930641_best.pth" \
    --seed ${SEED}
