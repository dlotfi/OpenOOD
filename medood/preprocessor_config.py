import argparse


class PreProcessorConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='PreProcessor')
        self.parser.add_argument('--base_dir',
                                 type=str,
                                 required=True,
                                 help='Base directory of the dataset')
        self.parser.add_argument('--output_dir',
                                 type=str,
                                 required=True,
                                 help='Output directory of the processed data')
        self.parser.add_argument('--num_samples',
                                 type=int,
                                 default=None,
                                 help='Number of samples to process')
        self.parser.add_argument('--seed',
                                 type=int,
                                 default=None,
                                 help='Random seed for reproducibility')
        self.parser.add_argument('--skip_existing',
                                 action='store_true',
                                 help='Skip re-processing existing files')
        self._args = None

    def parse_args(self):
        self._args = self.parser.parse_args()

    def __getattribute__(self, item):
        if item in ('parser', 'parse_args', '_args'):
            return super().__getattribute__(item)
        if self._args is None:
            raise AttributeError('You need to call parse_args() first')
        return self._args.__getattribute__(item)

    def __repr__(self):
        args_str = 'Parameters: \n'
        max_key_length = max(len(key) for key in vars(self._args).keys())
        args_str += '\n'.join(f'{key:<{max_key_length}} : {value}'
                              for key, value in vars(self._args).items())
        args_str += '\n' + '-' * 20
        return args_str


class PreProcessorBrainConfig(PreProcessorConfig):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('--use_gpu',
                                 action='store_true',
                                 help='Use GPU for brain extraction')
