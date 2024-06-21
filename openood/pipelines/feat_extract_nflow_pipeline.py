from torch import nn

from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.utils import setup_logger


class FullBackBoneNormalizingFlowNet(nn.Module):
    def __init__(self, backbone, nflow):
        super(FullBackBoneNormalizingFlowNet, self).__init__()
        self.backbone = backbone
        self.nflow = nflow

    def forward(self, x, return_feature=False):
        if not return_feature:
            raise ValueError('return_feature must be True')
        logits_cls, backbone_features = self.backbone(x, return_feature)
        return logits_cls, self.nflow.forward(backbone_features.flatten(1))


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

    def extract_features_test(self, net, evaluator, id_loader_dict,
                              ood_loader_dict):
        full_net = FullBackBoneNormalizingFlowNet(net['backbone'],
                                                  net['nflow'])
        # start extracting features
        print('\nStart Feature Extraction...', flush=True)
        print('\t ID test data...')
        evaluator.extract(net['backbone'], id_loader_dict['test'],
                          self.config.dataset.name)
        evaluator.extract(full_net, id_loader_dict['test'],
                          f'{self.config.dataset.name}_norm')
        split_types = ['nearood', 'farood']
        for ood_split in split_types:
            for dataset_name, ood_dl in ood_loader_dict[ood_split].items():
                print(f'\t OOD {ood_split}/{dataset_name} data...')
                evaluator.extract(net['backbone'], ood_dl, dataset_name)
                evaluator.extract(full_net, ood_dl, f'{dataset_name}_norm')
        print('\nComplete Feature Extraction!')

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

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
            self.extract_features_test(net, evaluator, id_loader_dict,
                                       ood_loader_dict)
        else:
            self.extract_features_train_val(net, evaluator, id_loader_dict,
                                            ood_loader_dict)
