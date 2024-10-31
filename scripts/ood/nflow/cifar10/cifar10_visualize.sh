#!/bin/bash
# sh scripts/ood/nflow/cifar10/cifar10_visualize.sh

SEED=0

# feature extraction
python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/nflow_resnet18_32x32.yml \
    configs/pipelines/test/feat_extract_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    --num_workers 8 \
    --network.pretrained True \
    --network.checkpoint "./results/cifar10_nflow_nflow_e100_lr0.0001_default/s${SEED}/best_nflow.ckpt" None \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint "./results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s${SEED}/best.ckpt" \
    --seed ${SEED}

# draw plots
python visualize.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    --score_dir "./results/cifar10_nflow_test_ood_ood_nflow_default/s${SEED}/ood/scores" \
    --feat_dir "./results/cifar10_nflow_feat_extract_nflow" \
    --out_dir "./results/cifar10_nflow_test_ood_ood_nflow_default/s${SEED}/ood" \
    --outlier_method auto \
    --seed ${SEED}
