import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np

from openood.utils import Config


class BaseAnalyzer(ABC):
    def __init__(self, config: Config, analyzer_config: Config):
        self.config = config
        self.analyzer_config = analyzer_config
        csid_split = ['csid'] \
            if self.config.analyzer.ood_scheme == 'fsood' else []
        self.id_splits = ['id'] + csid_split
        self.datasets = {'id': [self.config.dataset.name]}
        for split in csid_split + self.config.analyzer.ood_splits:
            if split in self.config.ood_dataset:
                self.datasets[split] = self.config.ood_dataset[split].datasets
            else:
                print(f'Split {split} not found in ood_dataset')

        model1_scores = {}
        model2_scores = {}
        for split_name, dataset_list in self.datasets.items():
            datasets_scores1 = self._load_scores(
                self.config.analyzer.model1_score_dir, dataset_list)
            datasets_scores2 = self._load_scores(
                self.config.analyzer.model2_score_dir, dataset_list)
            model1_scores.update({
                (split_name, dataset_name): dataset_scores
                for dataset_name, dataset_scores in datasets_scores1.items()
            })
            model2_scores.update({
                (split_name, dataset_name): dataset_scores
                for dataset_name, dataset_scores in datasets_scores2.items()
            })

        id_keys = [(split_name, dataset_name) for split_name in self.id_splits
                   for dataset_name in self.datasets[split_name]]
        self.model1_id_scores = np.hstack(
            [model1_scores.pop(key) for key in id_keys])
        self.model2_id_scores = np.hstack(
            [model2_scores.pop(key) for key in id_keys])
        self.model1_ood_scores_dict = model1_scores
        self.model2_ood_scores_dict = model2_scores

    @staticmethod
    def _load_scores(score_dir, datasets_names):
        scores = {}
        for dataset_name in datasets_names:
            feature_dict = np.load(
                os.path.join(score_dir, f'{dataset_name}.npz'))
            scores[dataset_name] = feature_dict['conf']
        return scores

    def _concatenate_scores(self, ood_scores1, ood_scores2):
        assert len(ood_scores1) == len(ood_scores2)
        scores1 = np.hstack([self.model1_id_scores, ood_scores1])
        scores2 = np.hstack([self.model2_id_scores, ood_scores2])
        true_labels = np.hstack(
            [np.ones_like(self.model1_id_scores),
             np.zeros_like(ood_scores1)])
        return scores1, scores2, true_labels

    def get_scores(self, analyze_type=None):
        if analyze_type == 'all':
            agg_model1_ood_scores = np.hstack(
                list(self.model1_ood_scores_dict.values()))
            agg_model2_ood_scores = np.hstack(
                list(self.model2_ood_scores_dict.values()))
            scores1, scores2, true_labels = self._concatenate_scores(
                agg_model1_ood_scores, agg_model2_ood_scores)
            yield (None, None), scores1, scores2, true_labels
        elif analyze_type == 'splits':
            for split_name in self.config.analyzer.ood_splits:
                agg_model1_ood_scores = np.hstack([
                    scores
                    for key, scores in self.model1_ood_scores_dict.items()
                    if key[0] == split_name
                ])
                agg_model2_ood_scores = np.hstack([
                    scores
                    for key, scores in self.model2_ood_scores_dict.items()
                    if key[0] == split_name
                ])
                scores1, scores2, true_labels = self._concatenate_scores(
                    agg_model1_ood_scores, agg_model2_ood_scores)
                yield (split_name, None), scores1, scores2, true_labels
        elif analyze_type == 'datasets':
            for key in self.model1_ood_scores_dict.keys():
                scores1, scores2, true_labels = self._concatenate_scores(
                    self.model1_ood_scores_dict[key],
                    self.model2_ood_scores_dict[key])
                yield key, scores1, scores2, true_labels

    @abstractmethod
    def analyze(self, true_labels: np.ndarray, model1_scores: np.ndarray,
                model2_scores: np.ndarray) -> Dict:
        pass

    def generate_output(self, results: List[Tuple[str, str, Dict]]):
        last_split_name = None
        have_dataset = False
        for split_name, dataset_name, result in results:
            if split_name is None and dataset_name is None:
                print(f'\n{" OVERALL ":=^50}', flush=True)
                self._print_results(result)
                continue
            if split_name != last_split_name:
                last_split_name = split_name
                have_dataset = False
                print(f'\n{" " + split_name.upper() + " ":=^50}', flush=True)
            if dataset_name is not None:
                print(f'\n{" " + dataset_name + " ":-^50}', flush=True)
                have_dataset = True
            elif have_dataset:
                print(f'\n{" SPLIT TOTAL ":-^50}', flush=True)
            self._print_results(result)

    @staticmethod
    def _print_results(results: Dict, indent=0):
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                continue
            elif isinstance(value, dict):
                print(' ' * indent + f'{key}:')
                BaseAnalyzer._print_results(value, indent + 4)
            else:
                print(' ' * indent + f'{key}: {value}', flush=True)

    def run(self):
        all_results = {}
        for analyze_type in self.analyzer_config.types:
            all_results.update({
                key: self.analyze(true_labels, scores1, scores2)
                for key, scores1, scores2, true_labels in self.get_scores(
                    analyze_type)
            })

        ordered_keys = []
        for split in self.config.analyzer.ood_splits:
            if split in self.config.ood_dataset:
                ordered_keys.extend([
                    (split, dataset)
                    for dataset in self.config.ood_dataset[split].datasets
                ])
                ordered_keys.append((split, None))
        ordered_keys.append((None, None))

        all_results = [(key[0], key[1], all_results[key])
                       for key in ordered_keys if key in all_results]

        self.generate_output(all_results)
