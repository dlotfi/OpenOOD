#!/bin/bash
# sh scripts/ood/vim/covid_test_ood_vim.sh

PYTHONPATH='.':$PYTHONPATH \

python main.py \
    --config configs/datasets/covid/covid.yml \
    configs/datasets/covid/covid_ood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/vim.yml \
    --num_workers 8 \
    --network.checkpoint "./results/covid_resnet18_224x224_base_e200_lr0.0001_default/s0/best.ckpt" \
    --mark 8 \
    --postprocessor.postprocessor_args.dim 256
