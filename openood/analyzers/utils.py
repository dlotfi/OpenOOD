from openood.utils import Config
from .bootstrapping import Bootstrapping
from .delongs_test import DelongsTest


def get_analyzer(config: Config):
    analyzers = {
        'delong': DelongsTest,
        'bootstrapping': Bootstrapping,
    }

    analyzers_config = config.analyzer
    analyzer_dict = {}
    for test_name in analyzers_config.analyzers:
        test_config = analyzers_config[test_name]
        analyzer_dict[test_name] = \
            analyzers[test_name](config, test_config)

    return analyzer_dict
