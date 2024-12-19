#!/bin/bash
# sh scripts/ood/vim/organamnist_test_ood_vim.sh

PYTHONPATH='.':$PYTHONPATH \

python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/networks/resnet18_28x28.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/vim.yml \
    --num_workers 8 \
    --network.checkpoint "./results/organamnist_resnet18_28x28/s0/resnet18_28_1.pth" \
    --network.checkpoint_key "net" \
    --postprocessor.postprocessor_args.dim 256
