import os
from abc import ABC, abstractmethod

import numpy as np
from typing import List
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import normalize as sk_normalize

from openood.utils import Config


class BaseVisualizer(ABC):
    def __init__(self, config: Config, plot_config: Config):
        self.config = config
        self.plot_config = plot_config
        csid_split = ['csid'] \
            if self.config.visualizer.ood_scheme == 'fsood' else []
        self.id_splits = ['id'] + csid_split
        self.datasets = {
            'id': [self.config.dataset.name],
        }
        for split in csid_split + self.config.visualizer.ood_splits:
            if split in self.config.ood_dataset:
                self.datasets[split] = self.config.ood_dataset[split].datasets
            else:
                print(f'Split {split} not found in ood_dataset')

    def get_label(self, split_name: str, max_length: int = 75):
        labels = {
            'nearood': 'Near OOD',
            'farood': 'Far OOD',
            'csid': 'Covariate-Shift ID',
            'id': 'ID'
        }
        label = labels[split_name] if split_name in labels else \
            ' '.join([w.capitalize() for w in split_name.split('_')])
        if split_name in self.datasets:
            dataset_names = ', '.join(self.datasets[split_name])
            if len(dataset_names) + len(label) > max_length:
                dataset_names = dataset_names[:max_length - len(label)] + \
                                ' ...'
            label += f' ({dataset_names})'

        return label

    def get_dataset_label(self, x: str, max_length: int = 75):
        return self.get_label(x, max_length) if x in self.id_splits else x

    @staticmethod
    def _remove_outlier_data(values, method='zscore', sigma=3.0):
        if method is None:
            keep_indices = np.ones(len(values), dtype=bool)
        elif method == 'zscore':
            keep_indices = abs(values - np.mean(values)) < \
                           sigma * np.std(values)
        elif method == 'iqr':
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            keep_indices = (values >= Q1 - sigma * IQR) & \
                           (values <= Q3 + sigma * IQR)
        elif method == 'mad':
            median = np.median(values)
            mad = np.median(abs(values - median))
            keep_indices = abs(values - median) < sigma * mad
        else:
            raise ValueError(f'Unknown outlier removal method: {method}')

        return keep_indices

    @staticmethod
    def _evaluate_data(values):
        # Center and scale the data
        values_mean = np.mean(np.array(values, np.float64))
        values_std = np.std(np.array(values, np.float64))
        if values_std == 0:  # Prevent division by zero
            return 0
        scaled_values = \
            (np.array(values, np.float64) - values_mean) / values_std

        # Calculate skewness and kurtosis
        skewness = abs(skew(scaled_values))
        # Adjust kurtosis to match the normal distribution by subtracting 3
        kurt = abs(kurtosis(scaled_values, fisher=False) - 3)
        # A lower value indicates data closer to normal distribution
        score = (skewness + kurt) / 2

        return np.inf if np.isnan(score) else score

    def remove_outliers(self, name: str, scores: np.ndarray, *arrays):
        method = self.plot_config.score_outlier_removal.method
        keep_range = self.plot_config.score_outlier_removal.keep_range
        sigma = self.plot_config.score_outlier_removal.sigma
        keep_ratio_threshold = \
            self.plot_config.score_outlier_removal.keep_ratio_threshold

        if method == 'range':
            assert len(keep_range) == 2, 'keep_range must have 2 values'
            keep_range = [
                np.inf if str(x).lower() == 'inf' else
                -np.inf if str(x).lower() == '-inf' else float(x)
                for x in keep_range
            ]
            best_indices = np.where((scores >= keep_range[0])
                                    & (scores <= keep_range[1]))[0]
        elif method != 'auto':
            best_indices = self._remove_outlier_data(scores, method, sigma)
        else:
            methods = ['zscore', 'iqr', 'mad']
            sigmas = np.arange(0.25, 4.5, 0.25)
            best_indices = np.ones(len(scores), dtype=bool)
            best_score = self._evaluate_data(scores[best_indices])
            best_method = None
            best_sigma = None
            for method in methods:
                for sigma in sigmas:
                    keep_indices = self._remove_outlier_data(
                        scores, method, sigma)
                    if np.sum(keep_indices) < \
                       keep_ratio_threshold * len(scores):
                        continue
                    score = self._evaluate_data(scores[keep_indices])
                    if score < best_score:
                        best_score = score
                        best_method = method
                        best_sigma = sigma
                        best_indices = keep_indices
            if best_method is not None:
                print(f'Best outlier removal method for {name}: '
                      f'{best_method} (sigma={best_sigma})')
            else:
                print(f'No suitable outlier removal method could '
                      f'be found for {name}.')

        result = [scores[best_indices]]
        for array in arrays:
            result.append(array[best_indices])
        if len(result) == 1:
            return result[0]
        return tuple(result)

    def load_scores(self, filenames: List[str], separate=False):
        score_dir = self.config.visualizer.score_dir
        score_list = []
        for filename in filenames:
            feature_dict = np.load(os.path.join(score_dir, filename))
            score_list.append(feature_dict['conf'])
        if not separate:
            score_list = np.hstack(score_list)
        return score_list

    def load_features(self,
                      filenames: List[str],
                      separate=False,
                      l2_normalize=False,
                      z_normalize=False):
        feat_dir = self.config.visualizer.feat_dir
        feat_list = []
        for filename in filenames:
            feature_dict = np.load(os.path.join(feat_dir, filename))
            feats = feature_dict['feat_list']
            print(f"Loaded '{filename}' feature size: {feats.shape}")
            if z_normalize:
                mean = np.mean(feats, axis=0)
                std = np.std(feats, axis=0)
                feats = (feats - mean) / (std + 1e-6)
                print('Features have bean z-score normalized '
                      'in each dimension')
            if l2_normalize:
                feats = sk_normalize(feats, norm='l2', axis=1)
                print('Features have bean l2-normalized.')
            feat_list.append(feats)
        if not separate:
            feat_list = np.hstack(feat_list)
        return feat_list

    @staticmethod
    def get_title_and_file_suffix(l2_normalize_feat, z_normalize_feat):
        title_suffixes = []
        file_suffixes = []
        if z_normalize_feat:
            title_suffixes.append(' Z-Score Normalized')
            file_suffixes.append('z')
        if l2_normalize_feat:
            title_suffixes.append(' L2-Normalized')
            file_suffixes.append('l2')
        title_suffix = ' &'.join(title_suffixes)
        file_suffix = '_'.join(file_suffixes)
        if file_suffix:
            file_suffix = '_' + file_suffix + '_normalized'
        return title_suffix, file_suffix

    @abstractmethod
    def draw(self):
        pass
