from openood.utils import Config

from .feat_extract_pipeline import FeatExtractPipeline
from .feat_extract_opengan_pipeline import FeatExtractOpenGANPipeline
from .feat_extract_nflow_pipeline import FeatExtractNormalizingFlowPipeline
from .finetune_pipeline import FinetunePipeline
from .test_acc_pipeline import TestAccPipeline
from .test_ad_pipeline import TestAdPipeline
from .test_nflow_pipeline import TestNormalizingFlowPipeline
from .test_ood_pipeline import TestOODPipeline
from .train_ad_pipeline import TrainAdPipeline
from .train_aux_pipeline import TrainARPLGANPipeline
from .train_med3d_pipeline import TrainMed3DPipeline
from .train_nflow_pipeline import TrainNormalizingFlowPipeline
from .train_oe_pipeline import TrainOEPipeline
# from .train_only_pipeline import TrainOpenGanPipeline
from .train_opengan_pipeline import TrainOpenGanPipeline
from .train_pipeline import TrainPipeline
from .test_ood_pipeline_aps import TestOODPipelineAPS
from .visualize_ood_pipeline import VisualizeOODPipeline
from .analyze_ood_pipeline import AnalyzeOODPipeline


def get_pipeline(config: Config):
    pipelines = {
        'train': TrainPipeline,
        'finetune': FinetunePipeline,
        'test_acc': TestAccPipeline,
        'feat_extract': FeatExtractPipeline,
        'feat_extract_opengan': FeatExtractOpenGANPipeline,
        'feat_extract_nflow': FeatExtractNormalizingFlowPipeline,
        'test_ood': TestOODPipeline,
        'test_nflow': TestNormalizingFlowPipeline,
        'test_ad': TestAdPipeline,
        'train_ad': TrainAdPipeline,
        'train_oe': TrainOEPipeline,
        'train_opengan': TrainOpenGanPipeline,
        'train_arplgan': TrainARPLGANPipeline,
        'test_ood_aps': TestOODPipelineAPS,
        'train_nflow': TrainNormalizingFlowPipeline,
        'train_med3d': TrainMed3DPipeline,
        'visualize_ood': VisualizeOODPipeline,
        'analyze_ood': AnalyzeOODPipeline,
    }

    return pipelines[config.pipeline.name](config)
