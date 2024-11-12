from openood.utils import Config

from .spectrum_visualizer import SpectrumVisualizer
from .tsne_nflow_visualizer import TSNEFlowVisualizer
from .tsne_visualizer import TSNEVisualizer
from .tsne_score_visualizer import TSNEScoreVisualizer


def get_visualizer(config: Config):
    visualizer = {
        'spectrum': SpectrumVisualizer,
        'tsne': TSNEVisualizer,
        'tsne_score': TSNEScoreVisualizer,
        'tsne_nflow': TSNEFlowVisualizer
    }

    visualizer_config = config.visualizer
    visualizer_dict = {}
    for plot in visualizer_config.plots:
        plot_config = visualizer_config[plot]
        visualizer_dict[plot] = visualizer[plot](config, plot_config)

    return visualizer_dict
