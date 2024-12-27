#!/bin/bash
# sh scripts/ood/nflow/organmnist/organamnist_analyze.sh

SEED=0
MARK="5_feats"
METHOD="vim"

# calculate statistical tests
python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/pipelines/test/analyze_ood.yml \
    --analyzer.ood_scheme ood \
    --analyzer.model1_score_dir "./results/organamnist_nflow_test_nflow_ood_nflow_${MARK}/s${SEED}/ood/scores" \
    --analyzer.model2_score_dir "./results/organamnist_resnet18_28x28_test_ood_ood_${METHOD}_default/s${SEED}/ood/scores" \
    --analyzer.ood_splits nearood farood \
    --seed ${SEED} \
    --mark ${MARK}_${METHOD}
