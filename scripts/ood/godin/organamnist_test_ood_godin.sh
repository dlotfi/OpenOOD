#!/bin/bash
# sh scripts/ood/godin/organamnist_test_ood_godin.sh

PYTHONPATH='.':$PYTHONPATH \

python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/networks/godin_net.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/godin.yml \
    --network.backbone.name resnet18_28x28 \
    --num_workers 8 \
    --network.checkpoint 'results/organamnist_godin_net_godin_e100_lr0.1_default/s0/best.ckpt' \
    --mark epoch_100
