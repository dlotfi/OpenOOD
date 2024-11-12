import os

from .tsne_visualizer import TSNEVisualizer


class TSNEFlowVisualizer(TSNEVisualizer):
    def plot_tsne(self):
        output_dir = self.config.output_dir
        normalize_feats = self.plot_config.normalize_feats
        n_samples = self.plot_config.n_samples

        feats_dict = {}
        feats_flow_dict = {}
        for split_name, dataset_list in self.datasets.items():
            feats = self.load_features([f'{d}.npz' for d in dataset_list],
                                       separate=True,
                                       normalize=normalize_feats)
            feats_flow = self.load_features(
                [f'{d}_flow.npz' for d in dataset_list], separate=True)
            feats, feats_flow = self.random_sample([feats, feats_flow],
                                                   array_names=dataset_list,
                                                   n_samples=n_samples)
            feats_dict[split_name] = feats
            feats_flow_dict[split_name] = feats_flow

        print(
            'Plotting t-SNE for features of the backbone '
            'and normalizing flow',
            flush=True)
        if normalize_feats:
            title = 't-SNE for Normalized Backbone Features of ' \
                    'ID and OOD Samples'
            output_path = os.path.join(output_dir,
                                       'tsne_features_normalized.png')
        else:
            title = 't-SNE for Backbone Features of ID and OOD Samples'
            output_path = os.path.join(output_dir, 'tsne_features.png')
        self.draw_tsne_plot(feats_dict, title, output_path, self.get_label)
        self.draw_tsne_plot(
            feats_flow_dict,
            't-SNE for Normalizing Flow Features of ID and OOD Samples',
            os.path.join(output_dir, 'tsne_features_flow.png'), self.get_label)
