#!/bin/bash
# sh scripts/ood/nflow/organmnist/organamnist_analyze.sh

SEED=0
MARK="final_feat"

# calculate statistical tests
python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/pipelines/test/analyze_ood.yml \
    --analyzer.ood_scheme ood \
    --analyzer.model1_score_dir "./results/organamnist_nflow_test_ood_ood_nflow_${MARK}/s${SEED}/ood/scores" \
    --analyzer.model2_score_dir "./results/organamnist_resnet18_28x28_test_ood_ood_vim_default/s${SEED}/ood/scores" \
    --analyzer.ood_splits nearood farood \
    --seed ${SEED} \
    --mark ${MARK}
