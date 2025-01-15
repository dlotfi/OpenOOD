#!/bin/bash
# sh scripts/ood/nflow/brats20_t1/brats20_t1_analyze.sh

SEED=0
MARK="5_feats"
METHOD="vim"

# calculate statistical tests
python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    configs/pipelines/test/analyze_ood.yml \
    --analyzer.ood_scheme fsood \
    --analyzer.model1_score_dir "./results/brats20_t1_nflow_test_nflow_ood_nflow_${MARK}/s${SEED}/fsood/scores" \
    --analyzer.model2_score_dir "./results/brats20_t1_resnet3d_18_test_ood_ood_${METHOD}_default/s${SEED}/fsood/scores" \
    --analyzer.ood_splits transformation_shift population_shift modality_shift diagnostic_shift organ_shift \
    --analyzer.model_names Ours ${METHOD^^} \
    --analyzer.bootstrapping.types all splits datasets \
    --seed ${SEED} \
    --mark ${MARK}_${METHOD}
