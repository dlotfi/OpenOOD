import torch
from torch import nn
from torch.nn.functional import adaptive_avg_pool2d


class FeatureConcatNetwork(nn.Module):
    def __init__(self, encoder, layers):
        super(FeatureConcatNetwork, self).__init__()
        self.encoder = encoder
        self.layers = layers

    def forward(self, x, return_feature=False):
        if not return_feature:
            return self.encoder(x, return_feature=False)
        logits_cls, features_list = self.encoder(x, return_feature_list=True)
        features_to_aggregate = [
            f for i, f in enumerate(features_list) if (i + 1) in self.layers
        ]
        concatenated_features = torch.cat([
            adaptive_avg_pool2d(f, (1, 1)).flatten(1)
            for f in features_to_aggregate
        ],
                                          dim=1)
        return logits_cls, concatenated_features
