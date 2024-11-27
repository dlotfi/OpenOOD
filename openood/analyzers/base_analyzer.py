import os
from abc import ABC, abstractmethod

import numpy as np

from openood.utils import Config


class BaseAnalyzer(ABC):
    def __init__(self, config: Config, analyzer_config: Config):
        self.config = config
        self.analyzer_config = analyzer_config
        csid_split = ['csid'] \
            if self.config.analyzer.ood_scheme == 'fsood' else []
        self.id_splits = ['id'] + csid_split
        self.datasets = {
            'id': [self.config.dataset.name],
        }
        for split in csid_split + self.config.analyzer.ood_splits:
            if split in self.config.ood_dataset:
                self.datasets[split] = self.config.ood_dataset[split].datasets
            else:
                print(f'Split {split} not found in ood_dataset')

        model1_scores = {}
        model2_scores = {}
        true_labels = {}
        for split_name, dataset_list in self.datasets.items():
            scores1 = self.load_scores(self.config.analyzer.model1_score_dir,
                                       [f'{d}.npz' for d in dataset_list])
            scores2 = self.load_scores(self.config.analyzer.model2_score_dir,
                                       [f'{d}.npz' for d in dataset_list])
            model1_scores[split_name] = scores1
            model2_scores[split_name] = scores2
            true_labels[split_name] = np.ones_like(scores1) \
                if split_name in self.id_splits else np.zeros_like(scores1)

        id_model1_scores = \
            [model1_scores.pop(split_name) for split_name in self.id_splits]
        id_model2_scores = \
            [model2_scores.pop(split_name) for split_name in self.id_splits]
        id_true_labels = \
            [true_labels.pop(split_name) for split_name in self.id_splits]

        ood_splits = list(model1_scores.keys())
        model1_scores['all'] = \
            [model1_scores[split_name] for split_name in ood_splits]
        model2_scores['all'] = \
            [model2_scores[split_name] for split_name in ood_splits]
        true_labels['all'] = \
            [true_labels[split_name] for split_name in ood_splits]

        self.model1_scores = {}
        self.model2_scores = {}
        self.true_labels = {}
        for split_name in model1_scores.keys():
            self.model1_scores[split_name] = \
                np.hstack(id_model1_scores + list(model1_scores[split_name]))
            self.model2_scores[split_name] = \
                np.hstack(id_model2_scores + list(model2_scores[split_name]))
            self.true_labels[split_name] = \
                np.hstack(id_true_labels + list(true_labels[split_name]))

    @staticmethod
    def load_scores(score_dir, filenames):
        scores = []
        for filename in filenames:
            feature_dict = np.load(os.path.join(score_dir, filename))
            scores.append(feature_dict['conf'])
        scores = np.hstack(scores)
        return scores

    @staticmethod
    def print_results(results, indent=0):
        for key, value in results.items():
            if isinstance(value, dict):
                print(' ' * indent + f'{key}:')
                BaseAnalyzer.print_results(value, indent + 4)
            elif isinstance(value, float):
                print(' ' * indent + f'{key}: {value:.4f}', flush=True)
            else:
                print(' ' * indent + f'{key}: {value}', flush=True)

    @abstractmethod
    def analyze(self, true_labels, model1_scores, model2_scores):
        pass

    def run(self):
        if 'aggregate' in self.analyzer_config.types:
            print(f'\n{" Aggregate ":-^50}', flush=True)
            results = self.analyze(self.true_labels['all'],
                                   self.model1_scores['all'],
                                   self.model2_scores['all'])
            self.print_results(results)
        if 'split' in self.analyzer_config.types:
            for split_name in self.config.analyzer.ood_splits:
                print(f'\n{" " + split_name + " ":-^50}', flush=True)
                results = self.analyze(self.true_labels[split_name],
                                       self.model1_scores[split_name],
                                       self.model2_scores[split_name])
                self.print_results(results)
