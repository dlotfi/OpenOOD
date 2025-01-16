import os

from .tsne_visualizer import TSNEVisualizer


class TSNEFlowVisualizer(TSNEVisualizer):
    def plot_tsne(self):
        output_dir = self.config.output_dir
        l2_normalize_feat = self.plot_config.l2_normalize_feat
        z_normalize_feat = self.plot_config.z_normalize_feat
        n_samples = self.plot_config.n_samples

        feats_dict = {}
        feats_flow_dict = {}
        for split_name, dataset_list in self.datasets.items():
            feats = self.load_features([f'{d}.npz' for d in dataset_list],
                                       separate=True,
                                       l2_normalize=l2_normalize_feat,
                                       z_normalize=z_normalize_feat)
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

        title_suffix, file_suffix = self.get_title_and_file_suffix(
            l2_normalize_feat, z_normalize_feat)
        title = f't-SNE for{title_suffix} Backbone Features of ' \
                'ID and OOD Samples'
        output_path = os.path.join(output_dir,
                                   f'tsne_features{file_suffix}.svg')
        self.draw_tsne_plot(feats_dict, title, output_path, self.get_label)
        self.draw_tsne_plot(
            feats_flow_dict,
            't-SNE for Normalizing Flow Features of ID and OOD Samples',
            os.path.join(output_dir, 'tsne_features_flow.svg'), self.get_label)
