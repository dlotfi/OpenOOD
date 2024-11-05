from openood.utils import Config

from .base_preprocessor import BasePreprocessor
from .cider_preprocessor import CiderPreprocessor
from .csi_preprocessor import CSIPreprocessor
from .cutpaste_preprocessor import CutPastePreprocessor
from .draem_preprocessor import DRAEMPreprocessor
from .augmix_preprocessor import AugMixPreprocessor
from .med3d_preprocessor import Med3DPreprocessor, Med3DTestPreprocessor
from .pixmix_preprocessor import PixMixPreprocessor
from .randaugment_preprocessor import RandAugmentPreprocessor
from .cutout_preprocessor import CutoutPreprocessor
from .test_preprocessor import TestStandardPreProcessor
from .palm_preprocessor import PALMPreprocessor


def get_preprocessor(config: Config, split):
    train_preprocessors = {
        'base': BasePreprocessor,
        'draem': DRAEMPreprocessor,
        'cutpaste': CutPastePreprocessor,
        'augmix': AugMixPreprocessor,
        'pixmix': PixMixPreprocessor,
        'randaugment': RandAugmentPreprocessor,
        'cutout': CutoutPreprocessor,
        'csi': CSIPreprocessor,
        'cider': CiderPreprocessor,
        'palm': PALMPreprocessor,
        'med3d': Med3DPreprocessor,
    }
    test_preprocessors = {
        'base': TestStandardPreProcessor,
        'draem': DRAEMPreprocessor,
        'cutpaste': CutPastePreprocessor,
        'med3d': Med3DTestPreprocessor,
    }

    if split == 'train':
        return train_preprocessors[config.preprocessor.name](config)
    else:
        try:
            return test_preprocessors[config.preprocessor.name](config)
        except KeyError:
            return test_preprocessors['base'](config)
