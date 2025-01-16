import os
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from .base_visualizer import BaseVisualizer
from openood.utils.vis_comm import save_fig_and_close


class SpectrumVisualizer(BaseVisualizer):
    @staticmethod
    def draw_histogram(scores_dict, output_path, title, log_scale, n_bins,
                       label_fn: Callable[[str], str]):
        plt.figure(figsize=(8, 3), dpi=300)
        for key, scores in scores_dict.items():
            plt.hist(scores,
                     n_bins,
                     density=True,
                     weights=np.ones(len(scores)) / len(scores),
                     alpha=0.5,
                     label=label_fn(key),
                     log=log_scale)
        plt.yticks([])
        plt.legend(loc='upper left', fontsize='small')
        plt.title(title)
        save_fig_and_close(output_path)

    def plot_spectrum(self):
        output_dir = self.config.output_dir
        log_scale = self.plot_config.score_log_scale
        n_bins = self.plot_config.n_bins

        scores_dict = {}
        for split_name, dataset_list in self.datasets.items():
            scores = self.load_scores([f'{d}.npz' for d in dataset_list])
            scores = self.remove_outliers(split_name, scores)
            scores_dict[split_name] = scores

        print('Plotting histogram of log-likelihood', flush=True)
        self.draw_histogram(scores_dict,
                            os.path.join(output_dir, 'spectrum.svg'),
                            'Log-Likelihood for ID and OOD Samples', log_scale,
                            n_bins, self.get_label)

    def plot_spectrum_split(self):
        output_dir = os.path.join(self.config.output_dir, 'split_plots')
        log_scale = self.plot_config.score_log_scale
        n_bins = self.plot_config.n_bins

        os.makedirs(output_dir, exist_ok=True)

        id_scores_dict = {}
        scores_dict = {}
        for split_name, dataset_list in self.datasets.items():
            if split_name in self.id_splits:
                dataset_list = [dataset_list]
            for dataset in dataset_list:
                dataset = [dataset] if type(dataset) is not list else dataset
                scores = self.load_scores([f'{d}.npz' for d in dataset])
                scores = self.remove_outliers(dataset[0], scores)
                if split_name in self.id_splits:
                    id_scores_dict[split_name] = scores
                else:
                    scores_dict.setdefault(split_name, {})[dataset[0]] = scores

        # Plot spectrum for all pairs of 'id' and one of the other datasets
        for split_name, datasets_scores in scores_dict.items():
            print(f'Plotting histogram of log-likelihood for {split_name}',
                  flush=True)
            combined_scores_dict = {**id_scores_dict, **datasets_scores}
            self.draw_histogram(
                combined_scores_dict,
                os.path.join(output_dir, f'spectrum_{split_name}.svg'),
                f'Log-Likelihood for ID and {split_name} Samples', log_scale,
                n_bins, self.get_dataset_label)

    def draw(self):
        if 'all' in self.plot_config.types:
            print(f'\n{" Overall ":-^50}', flush=True)
            self.plot_spectrum()
        if 'splits' in self.plot_config.types:
            print(f'\n{" Split ":-^50}', flush=True)
            self.plot_spectrum_split()
