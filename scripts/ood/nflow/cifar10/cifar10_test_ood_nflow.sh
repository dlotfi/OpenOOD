#!/bin/bash
# sh scripts/ood/nflow/cifar10/cifar10_test_ood_nflow.sh

SEED=0
python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/nflow_resnet18_32x32.yml \
    configs/pipelines/test/test_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/nflow.yml \
    --num_workers 8 \
    --network.pretrained True \
    --network.checkpoint "./results/cifar10_nflow_nflow_e100_lr0.0001_default/s0/best_nflow.ckpt" None \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint "./results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s${SEED}/best.ckpt" \
    --seed ${SEED}
