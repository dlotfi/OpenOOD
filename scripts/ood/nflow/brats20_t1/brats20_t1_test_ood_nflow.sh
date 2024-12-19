#!/bin/bash
# sh scripts/ood/nflow/brats20_t1/brats20_t1_test_ood_nflow.sh

SEED=0
MARK="final_feat"

# direct evaluation (without saving extracted features)
python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    configs/networks/nflow_resnet3d_18.yml \
    configs/pipelines/test/test_nflow.yml \
    configs/preprocessors/med3d_preprocessor.yml \
    configs/postprocessors/nflow.yml \
    --evaluator.ood_scheme ood \
    --evaluator.ood_splits transformation_shift population_shift modality_shift diagnostic_shift organ_shift \
    --network.pretrained True \
    --network.checkpoint "./results/brats20_t1_nflow_nflow_e100_lr0.0001_${MARK}/s${SEED}/best_nflow.ckpt" None \
    --network.nflow.l2_normalize True \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint "./results/brats20_t1_resnet3d_18_med3d_e100_lr0.0001_default/s${SEED}/best.ckpt" \
    --seed ${SEED} \
    --mark ${MARK}
