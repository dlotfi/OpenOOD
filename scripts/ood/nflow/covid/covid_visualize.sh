#!/bin/bash
# sh scripts/ood/nflow/covid/covid_visualize.sh

SEED=0

# feature extraction
python main.py \
    --config configs/datasets/covid/covid.yml \
    configs/datasets/covid/covid_ood.yml \
    configs/networks/nflow_resnet18_224x224.yml \
    configs/pipelines/test/feat_extract_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    --num_workers 8 \
    --network.pretrained True \
    --network.checkpoint "./results/covid_nflow_nflow_e100_lr0.0001_default/s${SEED}/best_nflow.ckpt" None \
    --network.nflow.l2_normalize True \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint "./results/covid_resnet18_224x224_base_e200_lr0.0001_default/s${SEED}/best.ckpt" \
    --seed ${SEED}

# draw plots
python visualize.py \
    --config configs/datasets/covid/covid.yml \
    configs/datasets/covid/covid_ood.yml \
    --score_dir "./results/covid_nflow_test_ood_ood_nflow_default/s${SEED}/ood/scores" \
    --feat_dir "./results/covid_nflow_feat_extract_nflow" \
    --out_dir "./results/covid_nflow_test_ood_ood_nflow_default/s${SEED}/ood" \
    --outlier_method auto \
    --seed ${SEED}
