#!/bin/bash
# sh scripts/ood/react/organamnist_test_ood_react.sh

PYTHONPATH='.':$PYTHONPATH \

python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/networks/react_net.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/react.yml \
    --network.pretrained False \
    --network.backbone.name resnet18_28x28 \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint "./results/organamnist_resnet18_28x28/s0/resnet18_28_1.pth" \
    --network.backbone.checkpoint_key "net" \
    --num_workers 8
