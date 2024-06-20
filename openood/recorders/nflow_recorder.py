import copy
import os
import time

import torch

from .base_recorder import BaseRecorder


class NormalizingFlowRecorder(BaseRecorder):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.save_dir = self.config.output_dir
        self.best_val_auroc = 0
        self.best_epoch_idx = 0

    def report(self, train_metrics, val_metrics):
        print('Epoch [{:03d}/{:03d}] | Time {:5d}s | Train Loss: {:.4f} | '
              'Val AUROC: {:.2f}\n'.format(train_metrics['epoch_idx'],
                                           self.config.optimizer.num_epochs,
                                           int(time.time() - self.begin_time),
                                           train_metrics['loss'],
                                           val_metrics['auroc']),
              flush=True)

    def save_model(self, net, val_metrics):
        nflow = net['nflow']
        epoch_idx = val_metrics['epoch_idx']

        try:
            nflow_wts = copy.deepcopy(nflow.module.state_dict())
        except AttributeError:
            nflow_wts = copy.deepcopy(nflow.state_dict())

        if self.config.recorder.save_all_models:
            save_pth = os.path.join(self.save_dir,
                                    'epoch-{}_nflow.ckpt'.format(epoch_idx))
            torch.save(nflow_wts, save_pth)

        if val_metrics['auroc'] >= self.best_val_auroc:
            self.best_epoch_idx = epoch_idx
            self.best_val_auroc = val_metrics['auroc']

            torch.save(nflow_wts,
                       os.path.join(self.output_dir, 'best_nflow.ckpt'))

    def summary(self):
        print('Training Completed! '
              'Best val AUROC: {:.6f} '
              'at epoch {:d}'.format(self.best_val_auroc, self.best_epoch_idx),
              flush=True)
