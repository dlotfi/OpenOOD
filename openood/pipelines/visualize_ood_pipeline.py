import random

import numpy as np
import torch

from openood.utils import setup_logger
from openood.visualizers import get_visualizer


class VisualizeOODPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # set random seed
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

        # get visualizer
        visualizer_dict = get_visualizer(self.config)

        # start calculating accuracy
        print('Start visualizing...', flush=True)
        for name, visualizer in visualizer_dict.items():
            print('\n' + u'\u2500' * 70, flush=True)
            print(f'Drawing {name}...\n', flush=True)
            visualizer.draw()
