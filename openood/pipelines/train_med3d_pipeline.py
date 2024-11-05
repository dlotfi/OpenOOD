from monai.utils import set_determinism

from openood.pipelines.train_pipeline import TrainPipeline


class TrainMed3DPipeline(TrainPipeline):
    def __init__(self, config) -> None:
        super(TrainMed3DPipeline, self).__init__(config)

    def run(self):
        set_determinism(seed=self.config.seed)
        # torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = False
        super().run()
