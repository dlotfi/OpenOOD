#!/bin/bash
# sh scripts/ood/nflow/organamnist_224/organamnist_224_visualize.sh

SEED=0

# feature extraction
python main.py \
    --config configs/datasets/medmnist/organamnist_224.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/networks/nflow_resnet18_224x224.yml \
    configs/pipelines/test/feat_extract_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    --num_workers 8 \
    --network.pretrained True \
    --network.checkpoint "./results/organamnist_nflow_nflow_e100_lr0.0001_default/s${SEED}/best_nflow.ckpt" None \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint "./results/organamnist_resnet18_224x224/s${SEED}/resnet18_224_1.pth" \
    --network.backbone.checkpoint_key "net" \
    --seed ${SEED}

# draw plots
python visualize.py \
    --config configs/datasets/medmnist/organamnist_224.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    --score_dir "./results/organamnist_nflow_test_ood_ood_nflow_default/s${SEED}/ood/scores" \
    --feat_dir "./results/organamnist_nflow_feat_extract_nflow" \
    --out_dir "./results/organamnist_nflow_test_ood_ood_nflow_default/s${SEED}/ood" \
    --outlier_method auto \
    --seed ${SEED}
