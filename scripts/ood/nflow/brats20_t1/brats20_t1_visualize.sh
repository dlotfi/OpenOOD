#!/bin/bash
# sh scripts/ood/nflow/brats20_t1/brats20_t1_visualize.sh

SEED=0
MARK1="5_feats"
MARK2=""
#MARK2="_z_l2"

# feature extraction
python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    configs/networks/nflow_resnet3d_18.yml \
    configs/pipelines/test/feat_extract_nflow.yml \
    configs/preprocessors/med3d_preprocessor.yml \
    --pipeline.extract_nflow False \
    --network.pretrained True \
    --network.checkpoint "./results/brats20_t1_nflow_nflow_e100_lr0.0001_${MARK1}${MARK2}/s${SEED}/best_nflow.ckpt" None \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint "./results/brats20_t1_resnet3d_18_med3d_e100_lr0.0001_default/s${SEED}/best.ckpt" \
    --dataset.z_normalize_feat True \
    --ood_dataset.z_normalize_feat True \
    --seed ${SEED} \
    --mark ${MARK1}

# draw plots
python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    configs/pipelines/test/visualize_ood.yml \
    --visualizer.ood_scheme fsood \
    --visualizer.score_dir "./results/brats20_t1_nflow_test_nflow_ood_nflow_${MARK1}${MARK2}/s${SEED}/fsood/scores" \
    --visualizer.feat_dir "./results/brats20_t1_nflow_feat_extract_nflow_${MARK1}/s${SEED}" \
    --visualizer.ood_splits transformation_shift population_shift modality_shift diagnostic_shift organ_shift \
    --visualizer.spectrum.types all splits \
    --visualizer.spectrum.score_outlier_removal.method range \
    --visualizer.spectrum.score_outlier_removal.keep_range 1000 inf \
    --visualizer.spectrum.n_bins 500 \
    --visualizer.tsne.types all splits \
    --visualizer.tsne.z_normalize_feat False \
    --visualizer.tsne.n_samples 100 \
    --visualizer.tsne_score.types all splits \
    --visualizer.tsne_score.z_normalize_feat False \
    --visualizer.tsne_score.score_outlier_removal.keep_range 1000 inf \
    --visualizer.tsne_score.n_samples 100 \
    --seed ${SEED} \
    --mark ${MARK1}${MARK2}
