import torch
from torch import nn


class FeatureConcatNetwork(nn.Module):
    def __init__(self, encoder, layers, n_spatial_dims=2):
        super(FeatureConcatNetwork, self).__init__()
        self.encoder = encoder
        self.layers = list(map(int, layers))
        if n_spatial_dims == 2:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif n_spatial_dims == 3:
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            raise ValueError(f'Unsupported n_spatial_dims: {n_spatial_dims}')

    def forward(self, x, return_feature=False):
        if not return_feature:
            return self.encoder(x, return_feature=False)
        logits_cls, features_list = self.encoder(x, return_feature_list=True)
        features_to_aggregate = [
            f for i, f in enumerate(features_list) if (i + 1) in self.layers
        ]
        concatenated_features = torch.cat(
            [self.avg_pool(f).flatten(1) for f in features_to_aggregate],
            dim=1)
        return logits_cls, concatenated_features
