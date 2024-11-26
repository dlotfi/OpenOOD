#!/bin/bash
# sh scripts/ood/nflow/brats20_t1/brats20_t1_statistical_test.sh

SEED=0
MARK="26"

# calculate statistical test (Delong's test)
python statistical_test.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    --score_dir_1 "./results/brats20_t1_nflow_test_ood_ood_nflow_${MARK}/s${SEED}/fsood/scores" \
    --score_dir_2 "./results/brats20_t1_resnet3d_18_test_ood_ood_vim_${MARK}/s${SEED}/fsood/scores" \
    --ood_scheme fsood \
    --ood_splits transformation_shift population_shift modality_shift diagnostic_shift organ_shift \
    --test_name delong
