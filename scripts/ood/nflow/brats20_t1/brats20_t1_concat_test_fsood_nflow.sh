#!/bin/bash
# sh scripts/ood/nflow/brats20_t1/brats20_t1_concat_test_fsood_nflow.sh

SEED=0
MARK="default"

python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    configs/networks/nflow_resnet3d_18_feat_concat.yml \
    configs/pipelines/test/test_nflow.yml \
    configs/preprocessors/med3d_preprocessor.yml \
    configs/postprocessors/nflow.yml \
    --num_workers 8 \
    --evaluator.ood_scheme fsood \
    --evaluator.ood_splits transformation_shift population_shift modality_shift diagnostic_shift organ_shift \
    --network.pretrained True \
    --network.checkpoint "./results/brats20_t1_nflow_nflow_e100_lr0.0001_${MARK}_feat_concat/s${SEED}/best_nflow.ckpt" None \
    --network.nflow.normalize_input True \
    --network.backbone.pretrained False \
    --network.backbone.encoder.pretrained True \
    --network.backbone.encoder.checkpoint "./results/brats20_t1_resnet3d_18_med3d_e100_lr0.0001_${MARK}/s${SEED}/best.ckpt" \
    --seed ${SEED} \
    --mark "${MARK}_feat_concat"
