#!/bin/bash
# sh scripts/ood/godin/organamnist_train_godin.sh

PYTHONPATH='.':$PYTHONPATH \

python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/networks/godin_net.yml \
    configs/pipelines/train/baseline.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/godin.yml \
    --dataset.train.drop_last True \
    --network.backbone.name resnet18_28x28 \
    --num_workers 8 \
    --trainer.name godin \
    --optimizer.num_epochs 100 \
    --merge_option merge \
    --seed 0
