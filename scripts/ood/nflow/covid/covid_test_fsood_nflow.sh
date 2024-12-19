#!/bin/bash
# sh scripts/ood/nflow/covid/covid_test_fsood_nflow.sh

SEED=0
python main.py \
    --config configs/datasets/covid/covid.yml \
    configs/datasets/covid/covid_fsood.yml \
    configs/networks/nflow_resnet18_224x224.yml \
    configs/pipelines/test/test_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/nflow.yml \
    --num_workers 8 \
    --evaluator.ood_scheme fsood \
    --network.pretrained True \
    --network.checkpoint "./results/covid_nflow_nflow_e100_lr0.0001_default/s${SEED}/best_nflow.ckpt" None \
    --network.nflow.l2_normalize True \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint "./results/covid_resnet18_224x224_base_e200_lr0.0001_default/s${SEED}/best.ckpt" \
    --seed ${SEED}
