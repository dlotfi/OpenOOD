from monai.utils import set_determinism

from openood.pipelines.train_pipeline import TrainPipeline


class TrainMed3DPipeline(TrainPipeline):
    def __init__(self, config) -> None:
        super(TrainMed3DPipeline, self).__init__(config)

    def run(self):
        set_determinism(seed=self.config.seed,
                        use_deterministic_algorithms=True)
        super().run()

    def report_test_metrics(self, test_metrics):
        report_str = '\nComplete Evaluation, Last accuracy {:.2f}'.format(
            100.0 * test_metrics['acc'])

        if 'f1' in test_metrics:
            report_str += ', F1 {:.2f}'.format(100.0 * test_metrics['f1'])
        if 'precision' in test_metrics:
            report_str += ', Precision {:.2f}'.format(
                100.0 * test_metrics['precision'])
        if 'recall' in test_metrics:
            report_str += ', Recall {:.2f}'.format(100.0 *
                                                   test_metrics['recall'])

        print(report_str, flush=True)
