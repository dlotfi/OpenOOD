#!/bin/bash
# sh scripts/ood/nflow/organamnist_224/organamnist_224_agg_visualize.sh

SEED=0

# feature extraction
python main.py \
    --config configs/datasets/medmnist/organamnist_224.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/networks/nflow_resnet18_224x224_feat_agg.yml \
    configs/pipelines/test/feat_extract_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    --num_workers 8 \
    --network.pretrained True \
    --network.checkpoint "./results/organamnist_nflow_featagg_nflow_e100_lr0.0001_default/s${SEED}/best_nflow.ckpt" \
                         "./results/organamnist_nflow_featagg_nflow_e100_lr0.0001_default/s${SEED}/best_feat_agg.ckpt" \
                         None \
    --network.backbone.pretrained False \
    --network.backbone.encoder.pretrained True \
    --network.backbone.encoder.checkpoint "./results/organamnist_resnet18_224x224/s${SEED}/resnet18_224_1.pth" \
    --network.backbone.encoder.checkpoint_key "net" \
    --seed ${SEED}

# draw plots
python visualize.py \
    --config configs/datasets/medmnist/organamnist_224.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    --score_dir "./results/organamnist_nflow_featagg_test_ood_ood_nflow_default/s${SEED}/ood/scores" \
    --feat_dir "./results/organamnist_nflow_featagg_feat_extract_nflow" \
    --out_dir "./results/organamnist_nflow_featagg_test_ood_ood_nflow_default/s${SEED}/ood" \
    --outlier_method auto \
    --seed ${SEED}
