#!/bin/bash
# sh scripts/ood/nflow/brats20_t1/brats20_t1_test_fsood_nflow.sh

SEED=0
python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    configs/networks/nflow_resnet3d_18.yml \
    configs/pipelines/test/test_nflow.yml \
    configs/preprocessors/med3d_preprocessor.yml \
    configs/postprocessors/nflow.yml \
    --num_workers 8 \
    --evaluator.ood_scheme fsood \
    --network.pretrained True \
    --network.checkpoint "./results/brats20_t1_nflow_nflow_e100_lr0.0001_default/s${SEED}/best_nflow.ckpt" None \
    --network.nflow.normalize_input True \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint "./results/brats20_t1_resnet3d_18_med3d_e100_lr0.0001_default/s${SEED}/best.ckpt" \
    --seed ${SEED}
