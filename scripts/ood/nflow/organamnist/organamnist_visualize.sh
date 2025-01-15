#!/bin/bash
# sh scripts/ood/nflow/organamnist/organamnist_visualize.sh

SEED=0
MARK1="5_feats"
MARK2=""
#MARK2="_z_l2"

# feature extraction
python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/networks/nflow_resnet18_28x28.yml \
    configs/pipelines/test/feat_extract_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint "./results/organamnist_nflow_nflow_e100_lr0.0001_${MARK1}${MARK2}/s${SEED}/best_nflow.ckpt" None \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint "./results/organamnist_resnet18_28x28/s${SEED}/resnet18_28_1.pth" \
    --network.backbone.checkpoint_key "net" \
    --seed ${SEED} \
    --mark ${MARK1}

# draw plots
python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/pipelines/test/visualize_ood.yml \
    --visualizer.ood_scheme ood \
    --visualizer.score_dir "./results/organamnist_nflow_test_nflow_ood_nflow_${MARK1}${MARK2}/s${SEED}/ood/scores" \
    --visualizer.feat_dir "./results/organamnist_nflow_feat_extract_nflow_${MARK1}/s${SEED}" \
    --visualizer.ood_splits nearood farood \
    --visualizer.spectrum.types all splits \
    --visualizer.spectrum.score_log_scale False \
    --visualizer.spectrum.score_outlier_removal.method range \
    --visualizer.spectrum.score_outlier_removal.keep_range -1000 inf \
    --visualizer.tsne.types all splits \
    --visualizer.tsne.z_normalize_feat False \
    --visualizer.tsne_score.types all splits \
    --visualizer.tsne_score.z_normalize_feat False \
    --seed ${SEED} \
    --mark ${MARK1}${MARK2}
