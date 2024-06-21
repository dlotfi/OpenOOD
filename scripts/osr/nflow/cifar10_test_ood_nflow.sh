#!/bin/bash
# sh scripts/osr/nflow/cifar10_test_ood_nflow.sh

# NOTE!!!!
# need to manually change the network checkpoint path (not backbone) in configs/pipelines/test/test_nflow.yml
# corresponding to different runs
SEED=0
python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/nflow.yml \
    configs/pipelines/test/test_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/nflow.yml \
    --num_workers 8 \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint "./results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s${SEED}/best.ckpt" \
    --seed ${SEED}
