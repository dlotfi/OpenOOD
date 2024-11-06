import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

import openood.utils.comm as comm
from openood.evaluators.base_evaluator import BaseEvaluator
from openood.postprocessors import BasePostprocessor
from openood.utils import Config


class Med3DEvaluator(BaseEvaluator):
    def __init__(self, config: Config):
        super().__init__(config)
        self.extra_metrics = [
            m.lower() for m in self.config.evaluator.get('extra_metrics', [])
        ]
        self.average = self.config.evaluator.get('average', 'weighted')

    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1):
        net.eval()

        loss_avg = 0.0
        correct = 0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for batch in tqdm(data_loader,
                              desc='Eval: ',
                              position=0,
                              leave=True,
                              disable=not comm.is_main_process()):
                # prepare data
                data = batch['data'].cuda()
                target = batch['label'].cuda()

                # forward
                output = net(data)

                loss = F.cross_entropy(output, target)

                # accuracy
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

                # collect all targets and predictions
                all_targets.extend(target.cpu().numpy())
                all_preds.extend(pred.cpu().numpy())

                # test loss average
                loss_avg += float(loss.data)

        loss = loss_avg / len(data_loader)
        acc = correct / len(data_loader.dataset)

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss)
        metrics['acc'] = self.save_metrics(acc)

        if 'f1' in self.extra_metrics:
            f1 = f1_score(all_targets, all_preds, average=self.average)
            metrics['f1'] = self.save_metrics(f1)
        if 'precision' in self.extra_metrics:
            precision = precision_score(all_targets,
                                        all_preds,
                                        average=self.average)
            metrics['precision'] = self.save_metrics(precision)
        if 'recall' in self.extra_metrics:
            recall = recall_score(all_targets, all_preds, average=self.average)
            metrics['recall'] = self.save_metrics(recall)

        return metrics
