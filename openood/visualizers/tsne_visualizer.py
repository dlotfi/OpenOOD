import os
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Callable
from .base_visualizer import BaseVisualizer
from openood.utils.vis_comm import save_fig_and_close


class TSNEVisualizer(BaseVisualizer):
    @staticmethod
    def _tsne_compute(feats_dict: Dict[str, np.array], n_components=50):
        start_time = time.time()
        # Concatenate all arrays in feats_dict
        all_feats = np.concatenate(list(feats_dict.values()))
        # Standardize the combined features (zero mean and unit variance)
        scaler = StandardScaler()
        all_feats = scaler.fit_transform(all_feats)
        # Apply PCA and TSNE
        if n_components < all_feats.shape[1]:
            pca = PCA(n_components)
            all_feats = pca.fit_transform(all_feats)
        tsne = TSNE(n_components=2, verbose=0, perplexity=40, max_iter=2000)
        tsne_pos_all = tsne.fit_transform(all_feats)
        # Split the transformed data back into separate arrays
        tsne_pos_dict = {}
        i = 0
        for key, feats in feats_dict.items():
            tsne_pos_dict[key] = tsne_pos_all[i:i + len(feats)]
            i += len(feats)

        hours, rem = divmod(time.time() - start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print('TSNE Computation Duration: {:0>2}:{:0>2}:{:05.2f}'.format(
            int(hours), int(minutes), seconds),
              flush=True)

        return tsne_pos_dict

    @staticmethod
    def random_sample(array_list: List[List[np.ndarray]],
                      array_names: List[str],
                      n_samples: int = None,
                      sample_rate: float = 1):
        # Ensure all arrays have the same shape
        first_list = array_list[0]
        assert all(len(first_list) == len(lst)
                   for lst in array_list[1:]) and \
               all(len(f) == len(a)  # 'f' and 'a' are np.ndarray
                   for lst in array_list[1:]
                   for f, a in zip(first_list, lst)), \
               'Corresponding arrays in array_list must have the same lengths'
        assert len(array_names) == len(first_list), \
               'The number of array_names must be the same as the ' + \
               'number of arrays in arrays_list'

        # check all array have the same shape
        all_sampled_indices = []
        total_samples = sum(len(a) for a in first_list)
        for array, name in zip(first_list, array_names):
            arr_total_samples = len(array)
            arr_n_samples = \
                int(sample_rate * arr_total_samples) \
                if n_samples is None else \
                int(n_samples * arr_total_samples / total_samples)
            if arr_n_samples > arr_total_samples:
                print(f'WARNING: Number of samples to sample from {name} '
                      'is larger than the total number of samples it has: '
                      f'{arr_n_samples} > {arr_total_samples}')
                arr_n_samples = arr_total_samples

            arr_sampled_indices = np.random.choice(arr_total_samples,
                                                   arr_n_samples,
                                                   replace=False)
            all_sampled_indices.append(arr_sampled_indices)

        sampled_array_list = []
        for lst in array_list:
            sampled_array = []
            for i, arr_sampled_indices in enumerate(all_sampled_indices):
                sampled_array.extend(lst[i][arr_sampled_indices].tolist())
            sampled_array_list.append(np.array(sampled_array))

        if len(sampled_array_list) == 1:
            return sampled_array_list[0]
        return tuple(sampled_array_list)

    @staticmethod
    def draw_tsne_plot(feats_dict: Dict[str, np.array], title, output_path,
                       label_fn: Callable[[str], str]):
        plt.figure(figsize=(8, 8), dpi=300)
        tsne_feats_dict = TSNEVisualizer._tsne_compute(feats_dict)
        for key, tsne_feats in tsne_feats_dict.items():
            plt.scatter(tsne_feats[:, 0],
                        tsne_feats[:, 1],
                        s=10,
                        alpha=0.5,
                        label=label_fn(key))
        plt.axis('off')
        plt.legend(loc='upper left', fontsize='small')
        plt.title(title)
        save_fig_and_close(output_path)

    def plot_tsne(self):
        output_dir = self.config.output_dir
        l2_normalize_feat = self.plot_config.l2_normalize_feat
        z_normalize_feat = self.plot_config.z_normalize_feat
        n_samples = self.plot_config.n_samples

        feats_dict = {}
        for split_name, dataset_list in self.datasets.items():
            feats = self.load_features([f'{d}.npz' for d in dataset_list],
                                       separate=True,
                                       l2_normalize=l2_normalize_feat,
                                       z_normalize=z_normalize_feat)
            feats = self.random_sample([feats],
                                       array_names=dataset_list,
                                       n_samples=n_samples)
            feats_dict[split_name] = feats

        print('Plotting t-SNE for features of the backbone', flush=True)
        title_suffix, file_suffix = self.get_title_and_file_suffix(
            l2_normalize_feat, z_normalize_feat)
        title = f't-SNE for{title_suffix} Backbone ' \
                'Features of ID and OOD Samples'
        output_path = os.path.join(output_dir,
                                   f'tsne_features{file_suffix}.svg')
        self.draw_tsne_plot(feats_dict, title, output_path, self.get_label)

    def plot_tsne_split(self):
        output_dir = os.path.join(self.config.output_dir, 'split_plots')
        l2_normalize_feat = self.plot_config.l2_normalize_feat
        z_normalize_feat = self.plot_config.z_normalize_feat
        n_samples = self.plot_config.n_samples

        os.makedirs(output_dir, exist_ok=True)

        id_feats_dict = {}
        feats_dict = {}
        for split_name, dataset_list in self.datasets.items():
            if split_name in self.id_splits:
                dataset_list = [dataset_list]
            for dataset in dataset_list:
                dataset = [dataset] if type(dataset) is not list else dataset
                feats = self.load_features([f'{d}.npz' for d in dataset],
                                           separate=True,
                                           l2_normalize=l2_normalize_feat,
                                           z_normalize=z_normalize_feat)
                feats = self.random_sample([feats],
                                           array_names=dataset,
                                           n_samples=n_samples)
                if split_name in self.id_splits:
                    id_feats_dict[split_name] = feats
                else:
                    feats_dict.setdefault(split_name, {})[dataset[0]] = feats

        # Plot t-SNE for all pairs of 'id' and one of the other datasets
        for split_name, datasets_feats in feats_dict.items():
            print(f'Plotting t-SNE for {split_name}', flush=True)
            combined_feats_dict = {**id_feats_dict, **datasets_feats}
            title_suffix, file_suffix = self.get_title_and_file_suffix(
                l2_normalize_feat, z_normalize_feat)
            title = f't-SNE for{title_suffix} Backbone Features of ' \
                    f'ID and {split_name} Samples'
            output_path = os.path.join(
                output_dir, f'tsne_features{file_suffix}_{split_name}.svg')
            self.draw_tsne_plot(combined_feats_dict, title, output_path,
                                self.get_dataset_label)

    def draw(self):
        if 'all' in self.plot_config.types:
            print(f'\n{" Overall ":-^50}', flush=True)
            self.plot_tsne()
        if 'splits' in self.plot_config.types:
            print(f'\n{" Split ":-^50}', flush=True)
            self.plot_tsne_split()
