import os
import time
from pathlib import Path
import torch
from .base_recorder import BaseRecorder


class Med3DRecorder(BaseRecorder):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.best_metric = 0.0
        self.best_metric_name = config.recorder.best_metric

    def report(self, train_metrics, val_metrics):
        report_str = ('\nEpoch {:03d} | Time {:5d}s | '
                      'Train Loss {:.4f} | Val Loss {:.3f} | '
                      'Val Acc {:.2f}').format(
                          train_metrics['epoch_idx'],
                          int(time.time() - self.begin_time),
                          train_metrics['loss'], val_metrics['loss'],
                          100.0 * val_metrics['acc'])

        if 'f1' in val_metrics:
            report_str += ' | Val F1 {:.2f}'.format(100.0 * val_metrics['f1'])
        if 'precision' in val_metrics:
            report_str += ' | Val Precision {:.2f}'.format(
                100.0 * val_metrics['precision'])
        if 'recall' in val_metrics:
            report_str += ' | Val Recall {:.2f}'.format(100.0 *
                                                        val_metrics['recall'])

        print(report_str, flush=True)

    def save_model(self, net, val_metrics):
        try:
            state_dict = net.module.state_dict()
        except AttributeError:
            state_dict = net.state_dict()

        if self.config.recorder.save_all_models:
            torch.save(
                state_dict,
                os.path.join(
                    self.output_dir,
                    'model_epoch{}.ckpt'.format(val_metrics['epoch_idx'])))

        # Determine the metric to use for saving the best model
        metric_value = val_metrics.get(self.best_metric_name, None)
        if metric_value is not None and metric_value >= self.best_metric:
            # delete the depreciated best model
            old_fname = 'best_epoch{}_{}_{}.ckpt'.format(
                self.best_epoch_idx, self.best_metric_name, self.best_metric)
            old_pth = os.path.join(self.output_dir, old_fname)
            Path(old_pth).unlink(missing_ok=True)

            # update the best model
            self.best_epoch_idx = val_metrics['epoch_idx']
            self.best_metric = metric_value
            torch.save(state_dict, os.path.join(self.output_dir, 'best.ckpt'))

            save_fname = 'best_epoch{}_{}_{}.ckpt'.format(
                self.best_epoch_idx, self.best_metric_name, self.best_metric)
            save_pth = os.path.join(self.output_dir, save_fname)
            torch.save(state_dict, save_pth)

        # save last path
        if val_metrics['epoch_idx'] == self.config.optimizer.num_epochs:
            save_fname = 'last_epoch{}_acc{:.4f}.ckpt'.format(
                val_metrics['epoch_idx'], val_metrics['acc'])
            save_pth = os.path.join(self.output_dir, save_fname)
            torch.save(state_dict, save_pth)

    def summary(self):
        print('Training Completed! '
              'Best {}: {:.2f} '
              'at epoch {:d}'.format(self.best_metric_name,
                                     100 * self.best_metric,
                                     self.best_epoch_idx),
              flush=True)
