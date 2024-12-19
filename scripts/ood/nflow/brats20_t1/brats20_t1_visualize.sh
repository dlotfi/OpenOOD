#!/bin/bash
# sh scripts/ood/nflow/brats20_t1/brats20_t1_visualize.sh

SEED=0
MARK="default"

# feature extraction
python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    configs/networks/nflow_resnet3d_18.yml \
    configs/pipelines/test/feat_extract_nflow.yml \
    configs/preprocessors/med3d_preprocessor.yml \
    --num_workers 8 \
    --network.pretrained True \
    --network.checkpoint "./results/brats20_t1_nflow_nflow_e100_lr0.0001_${MARK}/s${SEED}/best_nflow.ckpt" None \
    --network.nflow.l2_normalize True \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint "./results/brats20_t1_resnet3d_18_med3d_e100_lr0.0001_${MARK}/s${SEED}/best.ckpt" \
    --seed ${SEED} \
    --mark ${MARK}

# draw plots
python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    configs/pipelines/test/visualize_nflow_ood.yml \
    --visualizer.ood_scheme fsood \
    --visualizer.score_dir "./results/brats20_t1_nflow_test_ood_ood_nflow_${MARK}/s${SEED}/fsood/scores" \
    --visualizer.feat_dir "./results/brats20_t1_nflow_feat_extract_nflow_${MARK}" \
    --visualizer.ood_splits transformation_shift population_shift modality_shift diagnostic_shift organ_shift \
    --visualizer.spectrum.types aggregate split \
    --visualizer.tsne_nflow.types aggregate split \
    --visualizer.tsne_score.types aggregate split \
    --seed ${SEED} \
    --mark ${MARK}
