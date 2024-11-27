import random

import numpy as np
import torch

from openood.utils import setup_logger
from openood.analyzers import get_analyzer


class AnalyzeOODPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # set random seed
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

        # get analyzer
        analyzer_dict = get_analyzer(self.config)

        # start analyzing results
        print('Start analyzing...', flush=True)
        for name, analyzer in analyzer_dict.items():
            print('\n' + u'\u2500' * 70, flush=True)
            print(f'Performing {name} analysis ...\n', flush=True)
            analyzer.run()
