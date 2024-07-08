#!/bin/bash
# sh scripts/osr/nflow/cifar10_visualize.sh

SEED=0

# feature extraction
# NOTE!!!!
# need to manually change the network checkpoint path (not backbone) in configs/pipelines/test/feat_extract_nflow.yml
# corresponding to different runs
python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/nflow_resnet18_32x32.yml \
    configs/pipelines/test/feat_extract_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    --num_workers 8 \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint "./results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s${SEED}/best.ckpt" \
    --seed ${SEED}

# draw plots
python visualize.py \
    --score_dir "./results/cifar10_nflow_test_ood_ood_nflow_default/s${SEED}/ood/scores" \
    --feat_dir "./results/cifar10_nflow_feat_extract_nflow" \
    --out_dir "./results/cifar10_nflow_test_ood_ood_nflow_default/s${SEED}/ood" \
    --seed ${SEED}
