#!/bin/bash
# sh scripts/ood/msp/brats20_t1_test_fsood_react.sh

PYTHONPATH='.':$PYTHONPATH \

python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    configs/networks/react_net.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/med3d_preprocessor.yml \
    configs/postprocessors/react.yml \
    --num_workers 8 \
    --evaluator.ood_scheme fsood \
    --evaluator.ood_splits transformation_shift population_shift modality_shift diagnostic_shift organ_shift \
    --network.pretrained False \
    --network.backbone.name resnet3d_18 \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint "./results/brats20_t1_resnet3d_18_med3d_e100_lr0.0001_default/s0/best.ckpt"
