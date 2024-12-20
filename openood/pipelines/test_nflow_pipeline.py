import random
import time

import numpy as np
import torch

from openood.datasets import (get_dataloader, get_ood_dataloader,
                              get_feature_nflow_test_dataloaders)
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.postprocessors import get_postprocessor
from openood.utils import setup_logger


class TestNormalizingFlowPipeline:
    def __init__(self, config) -> None:
        self.config = config

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
        if self.config.dataset.get('feat_root', None) and \
           self.config.ood_dataset.get('feat_root', None):
            id_loader_dict, ood_loader_dict = \
                get_feature_nflow_test_dataloaders(
                    self.config.dataset, self.config.ood_dataset)
            print('Using feature-based dataloader', flush=True)
        else:
            if bool(self.config.dataset.z_normalize_feat) or \
               bool(self.config.ood.z_normalize_feat):
                raise ValueError(
                    'Cannot z-normalize features when features '
                    'not provided! Specify "feat_root" in config.')
            id_loader_dict = get_dataloader(self.config)
            ood_loader_dict = get_ood_dataloader(self.config)

        # init network
        net = get_network(self.config.network)

        # init ood evaluator
        evaluator = get_evaluator(self.config)

        # init ood postprocessor
        postprocessor = get_postprocessor(self.config)
        # setup for distance-based methods
        postprocessor.setup(net, id_loader_dict, ood_loader_dict)
        print('\n', flush=True)
        print(u'\u2500' * 70, flush=True)

        # start calculating accuracy
        print('\nStart evaluation...', flush=True)
        if self.config.evaluator.ood_scheme == 'fsood':
            acc_metrics = evaluator.eval_acc(
                net,
                id_loader_dict['test'],
                postprocessor,
                fsood=True,
                csid_data_loaders=ood_loader_dict['csid'])
        else:
            acc_metrics = evaluator.eval_acc(net, id_loader_dict['test'],
                                             postprocessor)
        print('\nAccuracy {:.2f}%'.format(100 * acc_metrics['acc']),
              flush=True)
        print(u'\u2500' * 70, flush=True)

        # start evaluating ood detection methods
        timer = time.time()
        if self.config.evaluator.ood_scheme == 'fsood':
            evaluator.eval_ood(net,
                               id_loader_dict,
                               ood_loader_dict,
                               postprocessor,
                               fsood=True)
        else:
            evaluator.eval_ood(net, id_loader_dict, ood_loader_dict,
                               postprocessor)
        print('Time used for eval_ood: {:.0f}s'.format(time.time() - timer))
        print('Completed!', flush=True)
