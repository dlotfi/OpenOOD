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


def get_label(split_name: str, datasets: List[str] = None):
    labels = {
        'nearood': 'Near OOD',
        'farood': 'Far OOD',
        'csid': 'Covariate-Shift ID',
        'id': 'ID',
        'flow_id': 'Flow ID',
        'flow_farood': 'Flow Far OOD',
        'flow_nearood': 'Flow Near OOD',
    }
    label = labels[split_name]
    if datasets is not None:
        label += f' ({", ".join(datasets)})'
    return label


def remove_outlier_data(values, method='zscore', sigma=3.0):
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


def evaluate_data(values):
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


def remove_outliers(scores,
                    features=None,
                    method='auto',
                    sigma=3.0,
                    keep_ratio_threshold=0.5):
    if method != 'auto':
        best_indices = remove_outlier_data(scores, method, sigma)
    else:
        methods = ['zscore', 'iqr', 'mad']
        sigmas = np.arange(0.25, 4.5, 0.25)
        best_indices = np.ones(len(scores), dtype=bool)
        best_score = evaluate_data(scores[best_indices])
        best_method = None
        best_sigma = None
        for method in methods:
            for sigma in sigmas:
                keep_indices = remove_outlier_data(scores, method, sigma)
                if np.sum(keep_indices) < keep_ratio_threshold * len(scores):
                    continue
                score = evaluate_data(scores[keep_indices])
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


def tsne_compute(feats_dict: Dict[str, array], n_components=50):
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
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=2000)
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


def load_scores(score_dir: str, datasets: List[str]):
    score_list = []
    for dataset in datasets:
        feature_dict = np.load(f'{score_dir}/{dataset}.npz')
        score_list.extend(feature_dict['conf'])
    score_list = np.array(score_list)
    return score_list


def plot_spectrum(datasets: dict,
                  score_dir: str,
                  out_dir: str,
                  outlier_method: str = None,
                  outlier_sigma=3.0,
                  keep_ratio_thresh=0.5,
                  log_scale=False):
    score_dir = os.path.normpath(score_dir)
    out_dir = os.path.normpath(out_dir)
    id_scores = load_scores(score_dir, datasets['id'])
    # csid_scores = load_scores(score_dir, datasets['csid'])
    nearood_scores = load_scores(score_dir, datasets['nearood'])
    farood_scores = load_scores(score_dir, datasets['farood'])

    id_scores = remove_outliers(id_scores,
                                method=outlier_method,
                                sigma=outlier_sigma,
                                keep_ratio_threshold=keep_ratio_thresh)
    # csid_scores = remove_outliers(csid_scores,
    #                               method=outlier_method,
    #                               threshold=outlier_sigma,
    #                               keep_ratio_threshold=keep_ratio_thresh)
    nearood_scores = remove_outliers(nearood_scores,
                                     method=outlier_method,
                                     sigma=outlier_sigma,
                                     keep_ratio_threshold=keep_ratio_thresh)
    # farood_scores = farood_scores[farood_scores > 1000]
    farood_scores = remove_outliers(farood_scores,
                                    method=outlier_method,
                                    sigma=outlier_sigma,
                                    keep_ratio_threshold=keep_ratio_thresh)

    scores_dict = {
        'farood': farood_scores,
        'nearood': nearood_scores,
        # 'csid': csid_scores,
        'id': id_scores
    }

    print('Plotting histogram of log-likelihood', flush=True)
    n_bins = 500
    plt.figure(figsize=(8, 3), dpi=300)
    for key, scores in scores_dict.items():
        plt.hist(scores,
                 n_bins,
                 density=True,
                 weights=np.ones(len(scores)) / len(scores),
                 alpha=0.5,
                 label=get_label(key, datasets[key]),
                 log=log_scale)

    plt.yticks([])
    plt.legend(loc='upper left', fontsize='small')
    plt.title('Log-Likelihood for ID and OOD Samples')
    plt.savefig(f'{out_dir}/spectrum.png', bbox_inches='tight')


def load_features(datasets: List[str],
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


def draw_tsne_plot(feats_dict, title, output_path, datasets):
    plt.figure(figsize=(8, 8), dpi=300)
    tsne_feats_dict = tsne_compute(feats_dict)
    for key, tsne_feats in tsne_feats_dict.items():
        plt.scatter(tsne_feats[:, 0],
                    tsne_feats[:, 1],
                    s=10,
                    alpha=0.5,
                    label=get_label(key, datasets.get(key, None)))
    plt.axis('off')
    plt.legend(loc='upper left', fontsize='small')
    plt.title(title)
    plt.savefig(output_path, bbox_inches='tight')


def plot_tsne(datasets: dict, feat_dir: str, out_dir: str,
              normalize_feats: bool):
    feat_dir = os.path.normpath(feat_dir)
    out_dir = os.path.normpath(out_dir)
    id_feats, id_feats_flow = load_features(datasets['id'],
                                            feat_dir,
                                            n_samples=1000)
    nearood_feats, nearood_feats_flow = load_features(datasets['nearood'],
                                                      feat_dir,
                                                      n_samples=1000)
    farood_feats, farood_feats_flow = load_features(datasets['farood'],
                                                    feat_dir,
                                                    n_samples=1000)
    if normalize_feats:
        feats_dict = {
            'farood': normalize(farood_feats, norm='l2', axis=1),
            'nearood': normalize(nearood_feats, norm='l2', axis=1),
            'id': normalize(id_feats, norm='l2', axis=1)
        }
    else:
        feats_dict = {
            'farood': farood_feats,
            'nearood': nearood_feats,
            'id': id_feats
        }
    feats_flow_dict = {
        'farood': farood_feats_flow,
        'nearood': nearood_feats_flow,
        'id': id_feats_flow
    }

    print('Plotting t-SNE for features', flush=True)
    if normalize_feats:
        draw_tsne_plot(
            feats_dict,
            't-SNE for Normalized Backbone Features of ID and OOD Samples',
            f'{out_dir}/tsne_features_normalized.png', datasets)
    else:
        draw_tsne_plot(feats_dict,
                       't-SNE for Backbone Features of ID and OOD Samples',
                       f'{out_dir}/tsne_features.png', datasets)
    draw_tsne_plot(
        feats_flow_dict,
        't-SNE for Normalizing Flow Features of ID and OOD Samples',
        f'{out_dir}/tsne_features_flow.png', datasets)


def draw_tsne_score_plot(feats_dict, scores_dict, title, output_path,
                         colored_id, log_scale, datasets):
    plt.figure(figsize=(10, 8), dpi=300)
    tsne_feats_dict = tsne_compute(feats_dict)
    excluded_id_key = 'id' if not colored_id else 'something_else'
    all_scores = np.concatenate([
        scores for key, scores in scores_dict.items() if key != excluded_id_key
    ])
    markers = {'farood': 's', 'nearood': '^', 'id': 'o'}
    cmap = plt.cm.rainbow
    if log_scale:
        min_score = all_scores.min()
        if min_score <= 0:
            all_scores = all_scores + abs(min_score) + 1
        norm = mcolors.LogNorm(vmin=all_scores.min(), vmax=all_scores.max())
    else:
        norm = mcolors.Normalize(vmin=all_scores.min(), vmax=all_scores.max())
    for i, (key, tsne_feats) in enumerate(tsne_feats_dict.items()):
        scores = scores_dict[key]
        if key == excluded_id_key:
            plt.scatter(tsne_feats[:, 0],
                        tsne_feats[:, 1],
                        s=10,
                        alpha=0.2,
                        marker=markers[key],
                        label=get_label(key, datasets[key]),
                        c='grey')
        else:
            plt.scatter(tsne_feats[:, 0],
                        tsne_feats[:, 1],
                        s=10,
                        alpha=0.5,
                        marker=markers[key],
                        label=get_label(key, datasets[key]),
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
    id_feats, _, id_scores = load_features(datasets['id'],
                                           feat_dir,
                                           score_dir,
                                           n_samples=1000)
    nearood_feats, _, nearood_scores = load_features(datasets['nearood'],
                                                     feat_dir,
                                                     score_dir,
                                                     n_samples=1000)
    farood_feats, _, farood_scores = load_features(datasets['farood'],
                                                   feat_dir,
                                                   score_dir,
                                                   n_samples=1000)

    id_scores, id_feats = remove_outliers(id_scores, id_feats, outlier_method,
                                          outlier_sigma, keep_ratio_thresh)
    nearood_scores, nearood_feats = remove_outliers(nearood_scores,
                                                    nearood_feats,
                                                    outlier_method,
                                                    outlier_sigma,
                                                    keep_ratio_thresh)
    farood_scores, farood_feats = remove_outliers(farood_scores, farood_feats,
                                                  outlier_method,
                                                  outlier_sigma,
                                                  keep_ratio_thresh)

    if normalize_feats:
        feats_dict = {
            'farood': normalize(farood_feats, norm='l2', axis=1),
            'nearood': normalize(nearood_feats, norm='l2', axis=1),
            'id': normalize(id_feats, norm='l2', axis=1)
        }
    else:
        feats_dict = {
            'farood': farood_feats,
            'nearood': nearood_feats,
            'id': id_feats,
        }
    scores_dict = {
        'farood': farood_scores,
        'nearood': nearood_scores,
        'id': id_scores,
    }

    print('Plotting t-SNE for features', flush=True)
    if normalize_feats:
        draw_tsne_score_plot(
            feats_dict, scores_dict,
            't-SNE for Normalized Backbone Features of ID and OOD Samples',
            f'{out_dir}/tsne_features_scores_normalized.png', colored_id,
            log_scale, datasets)
    else:
        draw_tsne_score_plot(
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
    opt, unknown_args = parser.parse_known_args()
    config = merge_configs(*[Config(path) for path in opt.config])
    # set random seed
    np.random.seed(opt.seed)
    # draw plots
    datasets = {
        # 'csid_datasets': config.fsood_dataset.csid.datasets,
        'farood': config.ood_dataset.farood.datasets,
        'nearood': config.ood_dataset.nearood.datasets,
        'id': [config.dataset.name]
    }
    plot_spectrum(datasets, opt.score_dir, opt.out_dir, opt.outlier_method,
                  opt.outlier_sigma, opt.outlier_thresh, opt.log_scale)
    plot_tsne(datasets, opt.feat_dir, opt.out_dir, opt.normalize_feats)
    plot_tsne_score(datasets, opt.feat_dir, opt.score_dir, opt.out_dir,
                    opt.normalize_feats, opt.outlier_method, opt.outlier_sigma,
                    opt.outlier_thresh, opt.log_scale)
