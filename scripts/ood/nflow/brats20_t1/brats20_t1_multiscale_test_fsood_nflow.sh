#!/bin/bash
# sh scripts/ood/nflow/brats20_t1/brats20_t1_concat_test_multiscale.sh

SEED=0
MARK1="5_feats" MARK2="symmetric"

# feature extraction
python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    configs/networks/nflow_resnet3d_18_multiscale.yml \
    configs/pipelines/test/feat_extract_nflow.yml \
    configs/preprocessors/med3d_preprocessor.yml \
    --pipeline.extract_nflow False \
    --network.pretrained True \
    --network.checkpoint "./results/brats20_t1_nflow_multiscale_nflow_e100_lr0.0001_${MARK1}_${MARK2}/s${SEED}/best_nflow.ckpt" None \
    --network.backbone.pretrained False \
    --network.backbone.encoder.pretrained True \
    --network.backbone.encoder.checkpoint "./results/brats20_t1_resnet3d_18_med3d_e100_lr0.0001_default/s${SEED}/best.ckpt" \
    --seed ${SEED} \
    --mark ${MARK1}

# evaluation
python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    configs/networks/nflow_resnet3d_18_multiscale.yml \
    configs/pipelines/test/test_nflow.yml \
    configs/postprocessors/nflow.yml \
    --evaluator.ood_scheme fsood \
    --evaluator.ood_splits transformation_shift population_shift modality_shift diagnostic_shift organ_shift \
    --dataset.feat_root "./results/brats20_t1_feat_concat_feat_extract_nflow_${MARK1}/s${SEED}" \
    --ood_dataset.feat_root "./results/brats20_t1_nflow_multiscale_feat_extract_nflow_${MARK1}/s${SEED}" \
    --network.pretrained True \
    --network.checkpoint "./results/brats20_t1_nflow_multiscale_nflow_e100_lr0.0001_${MARK1}_${MARK2}/s${SEED}/best_nflow.ckpt" None \
    --network.backbone.pretrained False \
    --seed ${SEED} \
    --mark "${MARK1}_${MARK2}"

# direct evaluation (without saving extracted features)
#python main.py \
#    --config configs/datasets/medood/brats20_t1.yml \
#    configs/datasets/medood/brats20_t1_fsood.yml \
#    configs/networks/nflow_resnet3d_18_multiscale.yml \
#    configs/pipelines/test/test_nflow.yml \
#    configs/preprocessors/med3d_preprocessor.yml \
#    configs/postprocessors/nflow.yml \
#    --evaluator.ood_scheme fsood \
#    --evaluator.ood_splits transformation_shift population_shift modality_shift diagnostic_shift organ_shift \
#    --network.pretrained True \
#    --network.checkpoint "./results/brats20_t1_nflow_multiscale_nflow_e100_lr0.0001_${MARK1}_${MARK2}/s${SEED}/best_nflow.ckpt" None \
#    --network.backbone.pretrained False \
#    --network.backbone.encoder.pretrained True \
#    --network.backbone.encoder.checkpoint "./results/brats20_t1_resnet3d_18_med3d_e100_lr0.0001_default/s${SEED}/best.ckpt" \
#    --seed ${SEED} \
#    --mark "${MARK1}_${MARK2}"
