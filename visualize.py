import argparse
import os.path
import time
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

id_datasets = ['cifar10']
# csid_datasets = ['cifar10c', 'imagenet10']
nearood_datasets = ['cifar100', 'tin']
farood_datasets = ['mnist', 'svhn', 'texture', 'place365']

colors = {
    'nearood': '#FFDEBF',
    'farood': '#FFC690',
    'csid': '#BFEBFF',
    'id': '#90B1C0'
}
labels = {
    'nearood': 'Near OOD',
    'farood': 'Far OOD',
    'csid': 'Covariate-Shift ID',
    'id': 'ID'
}


def remove_outliers(data, m=1):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def tsne_compute(x, n_components=50):
    start_time = time.time()
    if n_components < x.shape[1]:
        pca = PCA(n_components)
        x = pca.fit_transform(x)
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=2000)
    tsne_pos = tsne.fit_transform(x)

    hours, rem = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print('TSNE Computation Duration: {:0>2}:{:0>2}:{:05.2f}'.format(
        int(hours), int(minutes), seconds),
          flush=True)

    return tsne_pos


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

    font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 10}
    plt.rc('font', **font)
    plt.figure(figsize=(8, 3), dpi=200)

    n_bins = 500

    # csid_scores = remove_outliers(csid_scores)
    nearood_scores = remove_outliers(nearood_scores)
    farood_scores = remove_outliers(farood_scores)

    scores_dict = {
        'farood': farood_scores,
        'nearood': nearood_scores,
        # 'csid': csid_scores,
        'id': id_scores
    }
    for key, scores in scores_dict.items():
        plt.hist(scores,
                 n_bins,
                 density=True,
                 weights=np.ones(len(scores)) / len(scores),
                 facecolor=colors[key],
                 alpha=0.9,
                 label=labels[key])

    plt.yticks([])
    plt.legend(loc='upper left')
    plt.title('Log-Likelihood for ID and OOD Samples')
    plt.savefig(f'{out_dir}/spectrum.png', bbox_inches='tight')


def load_features(feat_dir: str, datasets: List[str], sample_rate: float):
    feat_list = []
    feat_norm_list = []
    for dataset in datasets:
        features = np.load(f'{feat_dir}/{dataset}.npz')['feat_list']
        num_samples = len(features)
        index_list = np.arange(num_samples)
        index_select = np.random.choice(index_list,
                                        int(sample_rate * num_samples),
                                        replace=False)
        feat_list.extend(features[index_select])
        features = np.load(f'{feat_dir}/{dataset}_norm.npz')['feat_list']
        feat_norm_list.extend(features[index_select])
    feat_list = np.array(feat_list)
    feat_norm_list = np.array(feat_norm_list)
    return feat_list, feat_norm_list


def plot_tsne(feat_dir: str, out_dir: str):
    feat_dir = os.path.normpath(feat_dir)
    out_dir = os.path.normpath(out_dir)
    id_feats, id_feats_norm = load_features(feat_dir, id_datasets, 0.1)
    nearood_feats, nearood_feats_norm = load_features(feat_dir,
                                                      nearood_datasets, 0.1)
    farood_feats, farood_feats_norm = load_features(feat_dir, farood_datasets,
                                                    0.1)

    font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 10}
    plt.rc('font', **font)
    plt.figure(figsize=(8, 8), dpi=200)

    feats_dict = {
        'farood': farood_feats,
        'nearood': nearood_feats,
        'id': id_feats
    }
    for key, feats in feats_dict.items():
        print(f'Plotting TSNE for features of the backbone on "{labels[key]}"',
              flush=True)
        tsne_feats = tsne_compute(feats)
        plt.scatter(tsne_feats[:, 0],
                    tsne_feats[:, 1],
                    c=colors[key],
                    s=5,
                    alpha=0.5,
                    label=labels[key])
    plt.axis('off')
    plt.legend(loc='upper left')
    plt.title('t-SNE for Backbone Features of ID and OOD Samples')
    plt.savefig(f'{out_dir}/tsne_features.png', bbox_inches='tight')

    plt.figure(figsize=(8, 8), dpi=200)

    feats_dict = {
        'farood': farood_feats_norm,
        'nearood': nearood_feats_norm,
        'id': id_feats_norm
    }
    for key, feats in feats_dict.items():
        print(
            f'Plotting TSNE for features of the normalizing '
            f'flow on "{labels[key]}"',
            flush=True)
        tsne_feats = tsne_compute(feats)
        plt.scatter(tsne_feats[:, 0],
                    tsne_feats[:, 1],
                    c=colors[key],
                    s=5,
                    alpha=0.5,
                    label=labels[key])
    plt.axis('off')
    plt.legend(loc='upper left')
    plt.title('t-SNE for Normalizing Flow Features of ID and OOD Samples')
    plt.savefig(f'{out_dir}/tsne_norm_features.png', bbox_inches='tight')


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
