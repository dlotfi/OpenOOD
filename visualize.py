import argparse
import os.path
import time
from typing import List, Dict

import numpy as np
import scienceplots
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from numpy import array
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, normalize

from openood.utils import Config
from openood.utils.config import merge_configs

__all__ = ('scienceplots', )
plt.style.use(['science', 'no-latex'])


def _get_label(split_name: str,
               datasets: List[str] = None,
               max_length: int = 75):
    labels = {
        'nearood': 'Near OOD',
        'farood': 'Far OOD',
        'csid': 'Covariate-Shift ID',
        'id': 'ID'
    }
    if split_name in labels:
        label = labels[split_name]
    elif split_name.startswith('flow') and split_name.split('_')[1] in labels:
        label = 'Flow ' + labels[split_name.split('_')[1]]
    else:
        # split by '_' and capitalize each word
        label = ' '.join([word.capitalize() for word in split_name.split('_')])
    if datasets is not None:
        dataset_names = ', '.join(datasets).replace('_', '-')
        if len(dataset_names) + len(label) > max_length:
            dataset_names = dataset_names[:max_length - len(label)] + ' ...'
        label += f' ({dataset_names})'

    return label


def _remove_outlier_data(values, method='zscore', sigma=3.0):
    if method is None:
        keep_indices = np.ones(len(values), dtype=bool)
    elif method == 'zscore':
        keep_indices = abs(values - np.mean(values)) < sigma * np.std(values)
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


def _evaluate_data(values):
    # Center and scale the data
    values_mean = np.mean(np.array(values, np.float64))
    values_std = np.std(np.array(values, np.float64))
    if values_std == 0:  # Prevent division by zero
        return 0
    scaled_values = (np.array(values, np.float64) - values_mean) / values_std

    # Calculate skewness and kurtosis
    skewness = abs(skew(scaled_values))
    # Adjust kurtosis to match the normal distribution by subtracting 3
    kurt = abs(kurtosis(scaled_values, fisher=False) - 3)
    # A lower value indicates data closer to normal distribution
    score = (skewness + kurt) / 2

    return np.inf if np.isnan(score) else score


def _remove_outliers(scores,
                     features=None,
                     method='auto',
                     sigma=3.0,
                     keep_ratio_threshold=0.5):
    if method != 'auto':
        best_indices = _remove_outlier_data(scores, method, sigma)
    else:
        methods = ['zscore', 'iqr', 'mad']
        sigmas = np.arange(0.25, 4.5, 0.25)
        best_indices = np.ones(len(scores), dtype=bool)
        best_score = _evaluate_data(scores[best_indices])
        best_method = None
        best_sigma = None
        for method in methods:
            for sigma in sigmas:
                keep_indices = _remove_outlier_data(scores, method, sigma)
                if np.sum(keep_indices) < keep_ratio_threshold * len(scores):
                    continue
                score = _evaluate_data(scores[keep_indices])
                if score < best_score:
                    best_score = score
                    best_method = method
                    best_sigma = sigma
                    best_indices = keep_indices
        if best_method is not None:
            print(f'Best outlier removal method: {best_method} '
                  f'(sigma={best_sigma})')
        else:
            print('No outlier has been removed.')

    if features is not None:
        return scores[best_indices], features[best_indices]
    return scores[best_indices]


def _tsne_compute(feats_dict: Dict[str, array], n_components=50):
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


def _load_scores(score_dir: str, datasets: List[str]):
    score_list = []
    for dataset in datasets:
        feature_dict = np.load(f'{score_dir}/{dataset}.npz')
        score_list.extend(feature_dict['conf'])
    score_list = np.array(score_list)
    return score_list


def plot_spectrum(datasets: dict,
                  score_dir: str,
                  out_dir: str,
                  cut_value: float = None,
                  outlier_method: str = None,
                  outlier_sigma=3.0,
                  keep_ratio_thresh=0.5,
                  log_scale=False):
    score_dir = os.path.normpath(score_dir)
    out_dir = os.path.normpath(out_dir)

    scores_dict = {}
    for split_name, dataset_list in datasets.items():
        scores = _load_scores(score_dir, dataset_list)
        if cut_value is not None:
            scores = scores[scores > cut_value]
        else:
            scores = _remove_outliers(scores,
                                      method=outlier_method,
                                      sigma=outlier_sigma,
                                      keep_ratio_threshold=keep_ratio_thresh)
        scores_dict[split_name] = scores

    print('Plotting histogram of log-likelihood', flush=True)
    n_bins = 500
    plt.figure(figsize=(8, 3), dpi=300)
    for key, scores in scores_dict.items():
        plt.hist(scores,
                 n_bins,
                 density=True,
                 weights=np.ones(len(scores)) / len(scores),
                 alpha=0.5,
                 label=_get_label(key, datasets[key]),
                 log=log_scale)

    plt.yticks([])
    plt.legend(loc='upper left', fontsize='small')
    plt.title('Log-Likelihood for ID and OOD Samples')
    cut_value_str = f'_cut{cut_value}' if cut_value is not None else ''
    plt.savefig(f'{out_dir}/spectrum{cut_value_str}.png', bbox_inches='tight')


def _load_features(datasets: List[str],
                   feat_dir: str,
                   score_dir: str = None,
                   n_samples: int = None,
                   sample_rate: float = 1):
    feat_list = []
    feat_flow_list = []
    score_list = []
    # Calculate total number of samples across all datasets
    total_samples = sum([
        len(np.load(f'{feat_dir}/{dataset}.npz')['feat_list'])
        for dataset in datasets
    ])
    for dataset in datasets:
        features = np.load(f'{feat_dir}/{dataset}.npz')['feat_list']
        total_samples_dataset = len(features)

        # Determine number of samples to select from this dataset
        n_samples_dataset = \
            int(sample_rate * total_samples_dataset) if n_samples is None \
            else int(n_samples * total_samples_dataset / total_samples)
        if n_samples_dataset > total_samples_dataset:
            print(f'WARNING: Number of samples to select '
                  f'({n_samples_dataset}) is greater than the total number '
                  f'of samples in the dataset ({total_samples_dataset}).')
            n_samples_dataset = total_samples_dataset

        index_select = np.random.choice(total_samples_dataset,
                                        n_samples_dataset,
                                        replace=False)
        feat_list.extend(features[index_select])

        features = np.load(f'{feat_dir}/{dataset}_flow.npz')['feat_list']
        feat_flow_list.extend(features[index_select])

        if score_dir is not None:
            feature_dict = np.load(f'{score_dir}/{dataset}.npz')
            score_list.extend(feature_dict['conf'][index_select])

    feat_list = np.array(feat_list)
    feat_flow_list = np.array(feat_flow_list)
    if score_dir is not None:
        score_list = np.array(score_list)
        return feat_list, feat_flow_list, score_list

    return feat_list, feat_flow_list


def _draw_tsne_plot(feats_dict, title, output_path, datasets):
    plt.figure(figsize=(8, 8), dpi=300)
    tsne_feats_dict = _tsne_compute(feats_dict)
    for key, tsne_feats in tsne_feats_dict.items():
        plt.scatter(tsne_feats[:, 0],
                    tsne_feats[:, 1],
                    s=10,
                    alpha=0.5,
                    label=_get_label(key, datasets.get(key, None)))
    plt.axis('off')
    plt.legend(loc='upper left', fontsize='small')
    plt.title(title)
    plt.savefig(output_path, bbox_inches='tight')


def plot_tsne(datasets: dict, feat_dir: str, out_dir: str,
              normalize_feats: bool):
    feat_dir = os.path.normpath(feat_dir)
    out_dir = os.path.normpath(out_dir)

    feats_dict = {}
    feats_flow_dict = {}

    for split_name, dataset_list in datasets.items():
        feats, feats_flow = _load_features(dataset_list,
                                           feat_dir,
                                           n_samples=1000)
        if normalize_feats:
            feats_dict[split_name] = normalize(feats, norm='l2', axis=1)
        else:
            feats_dict[split_name] = feats
        feats_flow_dict[split_name] = feats_flow

    print('Plotting t-SNE for features', flush=True)
    if normalize_feats:
        _draw_tsne_plot(
            feats_dict,
            't-SNE for Normalized Backbone Features of ID and OOD Samples',
            f'{out_dir}/tsne_features_normalized.png', datasets)
    else:
        _draw_tsne_plot(feats_dict,
                        't-SNE for Backbone Features of ID and OOD Samples',
                        f'{out_dir}/tsne_features.png', datasets)
    _draw_tsne_plot(
        feats_flow_dict,
        't-SNE for Normalizing Flow Features of ID and OOD Samples',
        f'{out_dir}/tsne_features_flow.png', datasets)


def _draw_tsne_score_plot(feats_dict, scores_dict, title, output_path,
                          colored_id, log_scale, datasets):
    plt.figure(figsize=(10, 8), dpi=300)
    tsne_feats_dict = _tsne_compute(feats_dict)
    all_scores = np.concatenate(
        [scores for key, scores in scores_dict.items()])
    cmap = plt.cm.rainbow
    if log_scale:
        min_score = all_scores.min()
        if min_score <= 0:
            all_scores = all_scores + abs(min_score) + 1
        norm = mcolors.LogNorm(vmin=all_scores.min(), vmax=all_scores.max())
    else:
        norm = mcolors.Normalize(vmin=all_scores.min(), vmax=all_scores.max())
    markers = [
        'o', 's', '^', 'D', 'v', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_'
    ]
    marker_dict = {
        key: markers[i % len(markers)]
        for i, key in enumerate(feats_dict.keys())
    }
    for key, tsne_feats in tsne_feats_dict.items():
        scores = scores_dict[key]
        marker = marker_dict[key]
        if key == 'id' and not colored_id:
            plt.scatter(tsne_feats[:, 0],
                        tsne_feats[:, 1],
                        s=10,
                        alpha=0.2,
                        marker=marker,
                        label=_get_label(key, datasets.get(key, None)),
                        c='grey')
        else:
            plt.scatter(tsne_feats[:, 0],
                        tsne_feats[:, 1],
                        s=10,
                        alpha=0.5,
                        marker=marker,
                        label=_get_label(key, datasets.get(key, None)),
                        c=scores,
                        cmap=cmap,
                        norm=norm)

    plt.colorbar(mappable=plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=plt.gca())
    plt.axis('off')
    legend = plt.legend(loc='upper left', fontsize='small')
    for handle in legend.legend_handles:
        handle.set_color('black')
    plt.title(title)
    plt.savefig(output_path, bbox_inches='tight')


def plot_tsne_score(datasets: dict,
                    feat_dir: str,
                    score_dir: str,
                    out_dir: str,
                    normalize_feats: bool,
                    outlier_method: str = None,
                    outlier_sigma=3.0,
                    keep_ratio_thresh=0.5,
                    log_scale=False,
                    colored_id=False):
    feat_dir = os.path.normpath(feat_dir)
    score_dir = os.path.normpath(score_dir)
    out_dir = os.path.normpath(out_dir)

    feats_dict = {}
    scores_dict = {}

    for split_name, dataset_list in datasets.items():
        feats, _, scores = _load_features(dataset_list,
                                          feat_dir,
                                          score_dir,
                                          n_samples=1000)
        scores, feats = _remove_outliers(scores, feats, outlier_method,
                                         outlier_sigma, keep_ratio_thresh)
        if normalize_feats:
            feats_dict[split_name] = normalize(feats, norm='l2', axis=1)
        else:
            feats_dict[split_name] = feats
        scores_dict[split_name] = scores

    print('Plotting t-SNE for features', flush=True)
    if normalize_feats:
        _draw_tsne_score_plot(
            feats_dict, scores_dict,
            't-SNE for Normalized Backbone Features of ID and OOD Samples',
            f'{out_dir}/tsne_features_scores_normalized.png', colored_id,
            log_scale, datasets)
    else:
        _draw_tsne_score_plot(
            feats_dict, scores_dict,
            't-SNE for Backbone Features of ID and OOD Samples',
            f'{out_dir}/tsne_features_scores.png', colored_id, log_scale,
            datasets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', nargs='+', required=True)
    parser.add_argument('--score_dir',
                        required=True,
                        help='path to the scores directory')
    parser.add_argument('--feat_dir',
                        required=True,
                        help='path to the features directory')
    parser.add_argument('--out_dir',
                        required=True,
                        help='path to the output directory')
    parser.add_argument('--log_scale',
                        action='store_true',
                        help='enable log scale for spectrum '
                        'and tsne-score plots')
    parser.add_argument('--outlier_method',
                        default=None,
                        help='the method for outlier removal '
                        '(auto, zscore, iqr, mad)')
    parser.add_argument('--outlier_sigma',
                        type=float,
                        default=3,
                        help='the sigma for outlier removal')
    parser.add_argument('--outlier_thresh',
                        type=float,
                        default=0.5,
                        help='the keep ratio threshold for outlier removal')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='seed for random number generation')
    parser.add_argument('--normalize_feats',
                        action='store_true',
                        help='Draw t-SNE plots with L2 normalized features')
    parser.add_argument('--splits',
                        nargs='+',
                        default=['nearood', 'farood'],
                        help='splits to visualize')
    parser.add_argument('--cut_value',
                        type=int,
                        default=None,
                        help='cut value for scores when drawing spectrum plot')
    parser.add_argument('--plots',
                        nargs='+',
                        choices=['spectrum', 'tsne', 'tsne_score'],
                        default=['spectrum', 'tsne', 'tsne_score'],
                        help='Specify which plots to draw: spectrum, tsne, '
                        'tsne_score. Default is all.')

    opt, unknown_args = parser.parse_known_args()
    config = merge_configs(*[Config(path) for path in opt.config])
    # set random seed
    np.random.seed(opt.seed)
    # draw plots
    datasets = {
        'id': [config.dataset.name],
    }
    for split in opt.splits:
        if split in config.ood_dataset:
            datasets[split] = config.ood_dataset[split].datasets
        else:
            print(f'Split {split} not found in ood_dataset')
    if 'spectrum' in opt.plots:
        plot_spectrum(datasets, opt.score_dir, opt.out_dir, opt.cut_value,
                      opt.outlier_method, opt.outlier_sigma,
                      opt.outlier_thresh, opt.log_scale)
    if 'tsne' in opt.plots:
        plot_tsne(datasets, opt.feat_dir, opt.out_dir, opt.normalize_feats)
    if 'tsne_score' in opt.plots:
        plot_tsne_score(datasets, opt.feat_dir, opt.score_dir, opt.out_dir,
                        opt.normalize_feats, opt.outlier_method,
                        opt.outlier_sigma, opt.outlier_thresh, opt.log_scale)
