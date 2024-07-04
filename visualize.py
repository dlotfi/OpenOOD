import argparse
import os.path
import time
from typing import List, Dict

import numpy as np
import scienceplots
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from numpy import array
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

__all__ = ('scienceplots', )
plt.style.use(['science', 'no-latex'])

id_datasets = ['cifar10']
# csid_datasets = ['cifar10c', 'imagenet10']
nearood_datasets = ['cifar100', 'tin']
farood_datasets = ['mnist', 'svhn', 'texture', 'place365']

labels = {
    'nearood': 'Near OOD',
    'farood': 'Far OOD',
    'csid': 'Covariate-Shift ID',
    'id': 'ID',
    'flow_id': 'Flow ID',
    'flow_farood': 'Flow Far OOD',
    'flow_nearood': 'Flow Near OOD',
}


def remove_outliers(data, m=3):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


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


def load_scores(score_dir: str, datasets: List[str], use_log: bool):
    score_list = []
    for dataset in datasets:
        feature_dict = np.load(f'{score_dir}/{dataset}.npz')
        score_list.extend(feature_dict['conf'])
    score_list = np.array(score_list)
    if use_log:
        score_list = np.log(score_list)
    return score_list


def plot_spectrum(score_dir: str, out_dir: str, use_log=False):
    score_dir = os.path.normpath(score_dir)
    out_dir = os.path.normpath(out_dir)
    id_scores = load_scores(score_dir, id_datasets, use_log)
    # csid_scores = load_scores(score_dir, csid_datasets, use_log)
    nearood_scores = load_scores(score_dir, nearood_datasets, use_log)
    farood_scores = load_scores(score_dir, farood_datasets, use_log)

    # csid_scores = remove_outliers(csid_scores, m=3)
    nearood_scores = remove_outliers(nearood_scores, m=2)
    farood_scores = remove_outliers(farood_scores, m=1)

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
                 label=labels[key])

    plt.yticks([])
    plt.legend(loc='upper left')
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


def draw_tsne_plot(feats_dict, title, output_path):
    plt.figure(figsize=(8, 8), dpi=300)
    tsne_feats_dict = tsne_compute(feats_dict)
    for key, tsne_feats in tsne_feats_dict.items():
        plt.scatter(tsne_feats[:, 0],
                    tsne_feats[:, 1],
                    s=10,
                    alpha=0.5,
                    label=labels[key])
    plt.axis('off')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.savefig(output_path, bbox_inches='tight')


def plot_tsne(feat_dir: str, out_dir: str, merge_plots=False):
    feat_dir = os.path.normpath(feat_dir)
    out_dir = os.path.normpath(out_dir)
    id_feats, id_feats_flow = load_features(id_datasets,
                                            feat_dir,
                                            n_samples=1000)
    nearood_feats, nearood_feats_flow = load_features(nearood_datasets,
                                                      feat_dir,
                                                      n_samples=1000)
    farood_feats, farood_feats_flow = load_features(farood_datasets,
                                                    feat_dir,
                                                    n_samples=1000)

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
    if merge_plots:
        feats_dict.update({('flow_' + k): v
                           for k, v in feats_flow_dict.items()})
        draw_tsne_plot(feats_dict, 't-SNE for Features of ID and OOD Samples',
                       f'{out_dir}/tsne_features_all.png')
    else:
        draw_tsne_plot(feats_dict,
                       't-SNE for Backbone Features of ID and OOD Samples',
                       f'{out_dir}/tsne_features.png')
        draw_tsne_plot(
            feats_flow_dict,
            't-SNE for Normalizing Flow Features of ID and OOD Samples',
            f'{out_dir}/tsne_features_flow.png')


def plot_tsne_score(feat_dir: str, score_dir: str, out_dir: str):
    feat_dir = os.path.normpath(feat_dir)
    out_dir = os.path.normpath(out_dir)
    id_feats, _, id_scores = load_features(id_datasets,
                                           feat_dir,
                                           score_dir,
                                           n_samples=1000)
    nearood_feats, _, nearood_scores = load_features(nearood_datasets,
                                                     feat_dir,
                                                     score_dir,
                                                     n_samples=1000)
    farood_feats, _, farood_scores = load_features(farood_datasets,
                                                   feat_dir,
                                                   score_dir,
                                                   n_samples=1000)
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

    plt.figure(figsize=(10, 8), dpi=300)
    tsne_feats_dict = tsne_compute(feats_dict)
    all_scores = np.concatenate(
        [scores for key, scores in scores_dict.items() if key != 'id'])
    markers = {'farood': 's', 'nearood': '^', 'id': 'o'}
    cmap = plt.cm.rainbow
    norm = mcolors.Normalize(vmin=all_scores.min(), vmax=all_scores.max())
    for i, (key, tsne_feats) in enumerate(tsne_feats_dict.items()):
        scores = scores_dict[key]
        if key == 'id':
            plt.scatter(tsne_feats[:, 0],
                        tsne_feats[:, 1],
                        s=10,
                        alpha=0.2,
                        marker=markers[key],
                        label=labels[key],
                        c='grey')
        else:
            plt.scatter(tsne_feats[:, 0],
                        tsne_feats[:, 1],
                        s=10,
                        alpha=0.5,
                        marker=markers[key],
                        label=labels[key],
                        c=scores,
                        cmap=cmap,
                        norm=norm)
    plt.colorbar(mappable=plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=plt.gca())
    plt.axis('off')
    legend = plt.legend(loc='upper left')
    for handle in legend.legend_handles:
        handle.set_color('black')
    plt.title('t-SNE for Backbone Features of ID and OOD Samples')
    plt.savefig(f'{out_dir}/tsne_features_scores.png', bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_dir',
                        help='path to the scores directory',
                        required=True)
    parser.add_argument('--feat_dir',
                        help='path to the features directory',
                        required=True)
    parser.add_argument('--out_dir',
                        help='path to the output directory',
                        required=True)
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Seed for random number generation',
                        required=False)
    opt = parser.parse_args()
    # set random seed
    np.random.seed(opt.seed)
    # draw plots
    plot_spectrum(opt.score_dir, opt.out_dir)
    plot_tsne(opt.feat_dir, opt.out_dir)
    plot_tsne_score(opt.feat_dir, opt.score_dir, opt.out_dir)
