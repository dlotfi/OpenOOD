from torch import nn


class LinearFeatureAggregateNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearFeatureAggregateNetwork, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, feats):
        return self.linear(feats)


def get_feature_aggregator(config):
    if config.type == 'linear':
        input_size = sum(config.layer_sizes)
        return LinearFeatureAggregateNetwork(input_size, config.output_size)
    else:
        raise Exception('Unexpected Feature Aggregator Network Type!')
