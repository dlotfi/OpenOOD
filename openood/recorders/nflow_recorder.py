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

    def _save_subnet(self, subnet, subnet_name, epoch_idx=None):
        if subnet is None:
            return
        try:
            subnet_wts = copy.deepcopy(subnet.module.state_dict())
        except AttributeError:
            subnet_wts = copy.deepcopy(subnet.state_dict())

        if epoch_idx is None:
            save_pth = os.path.join(self.output_dir,
                                    f'best_{subnet_name}.ckpt')
        else:
            save_pth = os.path.join(self.save_dir,
                                    f'epoch-{epoch_idx}_{subnet_name}.ckpt')
        torch.save(subnet_wts, save_pth)

    def save_model(self, net, val_metrics):
        nflow = net['nflow']
        feat_agg = net.get('feat_agg', None)
        epoch_idx = val_metrics['epoch_idx']

        if self.config.recorder.save_all_models:
            self._save_subnet(nflow, 'nflow', epoch_idx)
            self._save_subnet(feat_agg, 'feat_agg', epoch_idx)
        elif self.config.recorder.save_last_model:
            self._save_subnet(nflow, 'nflow', 'last')
            self._save_subnet(feat_agg, 'feat_agg', 'last')

        if val_metrics['auroc'] >= self.best_val_auroc:
            self.best_epoch_idx = epoch_idx
            self.best_val_auroc = val_metrics['auroc']
            self._save_subnet(nflow, 'nflow')
            self._save_subnet(feat_agg, 'feat_agg')

    def summary(self):
        print('Training Completed! '
              'Best val AUROC: {:.6f} '
              'at epoch {:d}'.format(self.best_val_auroc, self.best_epoch_idx),
              flush=True)
