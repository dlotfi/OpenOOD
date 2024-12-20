import random

import numpy as np
import torch
from torch import nn

from openood.datasets import (get_dataloader, get_ood_dataloader,
                              get_feature_nflow_test_dataloaders)
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.utils import setup_logger


class FullNormalizingFlowNet(nn.Module):
    def __init__(self, net):
        super(FullNormalizingFlowNet, self).__init__()
        self.feat_agg = net.get('feat_agg', nn.Identity())
        self.nflow = net['nflow']

    def forward(self, x, return_feature=False):
        if not return_feature:
            raise ValueError('return_feature must be True')
        logits_cls = torch.ones_like(x)  # dummy predictions
        feats = self.feat_agg(x.flatten(1))
        nflow_feats = self.nflow.inverse(feats)
        return logits_cls, nflow_feats


class FeatExtractNormalizingFlowPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def extract_features_train_val(self, net, evaluator, id_loader_dict,
                                   ood_loader_dict):
        # sanity check on id val accuracy
        print('\nStart evaluation on ID val data...', flush=True)
        test_metrics = evaluator.eval_acc(net, id_loader_dict['val'])
        print('\nComplete Evaluation, accuracy {:.2f}%'.format(
            100 * test_metrics['acc']),
              flush=True)

        # start extracting features
        print('\nStart Feature Extraction...', flush=True)
        print('\t ID training data...')
        evaluator.extract(net, id_loader_dict['train'], 'id_train')

        print('\t ID val data...')
        evaluator.extract(net, id_loader_dict['val'], 'id_val')

        print('\t OOD val data...')
        evaluator.extract(net, ood_loader_dict['val'], 'ood_val')
        print('\nComplete Feature Extraction!')

    def extract_backbone_features_test(self, net, evaluator, id_loader_dict,
                                       ood_loader_dict):
        backbone = net if 'backbone' not in net else net['backbone']
        # start extracting features
        print('\nStart Backbone Feature Extraction...', flush=True)
        print('\t ID test data...')
        evaluator.extract(backbone, id_loader_dict['test'],
                          self.config.dataset.name)
        if 'csid' in ood_loader_dict:
            for dataset_name, csid_dl in ood_loader_dict['csid'].items():
                print(f'\t CSID {dataset_name} data...')
                evaluator.extract(backbone, csid_dl, dataset_name)
        split_types = ood_loader_dict.keys() - {'csid', 'val'}
        for ood_split in split_types:
            for dataset_name, ood_dl in ood_loader_dict[ood_split].items():
                print(f'\t {ood_split.upper()} {dataset_name} data...')
                evaluator.extract(backbone, ood_dl, dataset_name)
        print('\nComplete Backbone Feature Extraction!')

    def extract_nflow_features_test(self, net, evaluator, id_loader_dict,
                                    ood_loader_dict):
        full_nflow = FullNormalizingFlowNet(net)
        # start extracting features
        print('\nStart Flow Feature Extraction...', flush=True)
        print('\t ID test data...')
        evaluator.extract(full_nflow, id_loader_dict['test'],
                          f'{self.config.dataset.name}_flow')
        if 'csid' in ood_loader_dict:
            for dataset_name, csid_dl in ood_loader_dict['csid'].items():
                print(f'\t CSID {dataset_name} data...')
                evaluator.extract(full_nflow, csid_dl, f'{dataset_name}_flow')
        split_types = ood_loader_dict.keys() - {'csid', 'val'}
        for ood_split in split_types:
            for dataset_name, ood_dl in ood_loader_dict[ood_split].items():
                print(f'\t {ood_split.upper()} {dataset_name} data...')
                evaluator.extract(full_nflow, ood_dl, f'{dataset_name}_flow')
        print('\nComplete Feature Extraction!')

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # set random seed
        try:
            from monai.utils import set_determinism
            set_determinism(seed=self.config.seed,
                            use_deterministic_algorithms=True)
        except ImportError:
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)
            torch.use_deterministic_algorithms(True)

        # get dataloader
        id_loader_dict = get_dataloader(self.config)
        ood_loader_dict = get_ood_dataloader(self.config)
        if self.config.pipeline.extract_target != 'test':
            assert 'train' in id_loader_dict
            assert 'val' in id_loader_dict
            assert 'val' in ood_loader_dict
        else:
            assert 'test' in id_loader_dict

        # init network
        net = get_network(self.config.network)

        # init evaluator
        evaluator = get_evaluator(self.config)

        if self.config.pipeline.extract_target == 'test':
            if self.config.pipeline.extract_backbone:
                self.extract_backbone_features_test(net, evaluator,
                                                    id_loader_dict,
                                                    ood_loader_dict)
            if self.config.pipeline.extract_nflow:
                self.config.ood_dataset.feat_root = self.config.output_dir
                id_loader_dict, ood_loader_dict = \
                    get_feature_nflow_test_dataloaders(
                        dataset_config=self.config.dataset,
                        ood_dataset_config=self.config.ood_dataset,
                        load_train_val=False)
                self.extract_nflow_features_test(net, evaluator,
                                                 id_loader_dict,
                                                 ood_loader_dict)
        else:
            self.extract_features_train_val(net, evaluator, id_loader_dict,
                                            ood_loader_dict)
