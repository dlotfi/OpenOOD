from normflows.nets import MLP


def get_feature_aggregator(config):
    if config.type == 'mlp':
        input_size = sum(config.layer_sizes)
        mlp1 = MLP([input_size, config.output_size])
        # mlp2 = MLP([input_size, input_size, config.output_size])
        # mlp3 = MLP([input_size, input_size * 2, config.output_size])
        # mlp4 = MLP([input_size, config.output_size, config.output_size])
        # mlp5 = MLP([input_size, input_size, config.output_size,
        #             config.output_size])
        # print(mlp1)
        return mlp1
    else:
        raise Exception('Unexpected Feature Aggregator Network Type!')
