#!/bin/bash
# sh scripts/ood/nflow/organamnist_plus_224/organamnist_plus_224_visualize.sh

SEED=0

# feature extraction
python main.py \
    --config configs/datasets/medmnist_plus/organamnist_224.yml \
    configs/datasets/medmnist_plus/organamnist_224_ood.yml \
    configs/networks/nflow_resnet18_224x224.yml \
    configs/pipelines/test/feat_extract_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    --num_workers 8 \
    --dataset.normalization_type imagenet \
    --network.pretrained True \
    --network.checkpoint "./results/organamnist_224_nflow_nflow_e100_lr0.0001_default/s${SEED}/best_nflow.ckpt" None \
    --network.nflow.l2_normalize True \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint "results/organamnist_224_resnet18_224/organamnist_224_endToEnd_resnet18_s9930641_best.pth" \
    --seed ${SEED}

# draw plots
python visualize.py \
    --config configs/datasets/medmnist_plus/organamnist_224.yml \
    configs/datasets/medmnist_plus/organamnist_224_ood.yml \
    --dataset.normalization_type imagenet \
    --score_dir "./results/organamnist_224_nflow_test_ood_ood_nflow_default/s${SEED}/ood/scores" \
    --feat_dir "./results/organamnist_224_nflow_feat_extract_nflow" \
    --out_dir "./results/organamnist_224_nflow_test_ood_ood_nflow_default/s${SEED}/ood" \
    --outlier_method auto \
    --normalize_feats \
    --seed ${SEED}
