#!/bin/bash
# sh scripts/ood/nflow/brats20_t1/brats20_t1_visualize.sh

SEED=0

# feature extraction
python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    configs/networks/nflow_resnet3d_18.yml \
    configs/pipelines/test/feat_extract_nflow.yml \
    configs/preprocessors/med3d_preprocessor.yml \
    --num_workers 8 \
    --network.pretrained True \
    --network.checkpoint "./results/brats20_t1_nflow_nflow_e100_lr0.0001_default/s${SEED}/best_nflow.ckpt" None \
    --network.nflow.normalize_input True \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint "./results/brats20_t1_resnet3d_18_med3d_e100_lr0.0001_default/s${SEED}/best.ckpt" \
    --seed ${SEED}

# draw plots
python visualize.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    --score_dir "./results/brats20_t1_nflow_test_ood_ood_nflow_default/s${SEED}/fsood/scores" \
    --feat_dir "./results/brats20_t1_nflow_feat_extract_nflow_default" \
    --out_dir "./results/brats20_t1_nflow_test_ood_ood_nflow_default/s${SEED}/fsood" \
    --normalize_feats \
    --ood_scheme fsood \
    --seed ${SEED}
