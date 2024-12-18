#!/bin/bash
# sh scripts/ood/msp/brats20_t1_test_fsood_vim.sh

PYTHONPATH='.':$PYTHONPATH \

python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    configs/networks/resnet3d_18.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/med3d_preprocessor.yml \
    configs/postprocessors/vim.yml \
    --num_workers 8 \
    --evaluator.ood_scheme fsood \
    --evaluator.ood_splits transformation_shift population_shift modality_shift diagnostic_shift organ_shift \
    --network.checkpoint "./results/brats20_t1_resnet3d_18_med3d_e100_lr0.0001_default/s0/best.ckpt" \
    --postprocessor.postprocessor_args.dim 256
