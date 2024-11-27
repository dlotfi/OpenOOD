#!/bin/bash
# sh scripts/ood/nflow/brats20_t1/brats20_t1_analyze.sh

SEED=0
MARK="26"

# calculate statistical tests
python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    configs/pipelines/test/analyze_ood.yml \
    --analyzer.ood_scheme fsood \
    --analyzer.model1_score_dir "./results/brats20_t1_nflow_test_ood_ood_nflow_${MARK}/s${SEED}/fsood/scores" \
    --analyzer.model2_score_dir "./results/brats20_t1_resnet3d_18_test_ood_ood_vim_${MARK}/s${SEED}/fsood/scores" \
    --analyzer.ood_splits transformation_shift population_shift modality_shift diagnostic_shift organ_shift \
    --seed ${SEED} \
    --mark ${MARK}
