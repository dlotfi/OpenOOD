import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import openood.utils.comm as comm


class NormalizingFlowTrainer:
    def __init__(self, net, feat_loader, config) -> None:
        self.config = config
        self.net = net
        self.nflow = net['nflow']
        self.feat_loader = feat_loader

        for p in self.net['backbone'].parameters():
            p.requires_grad = False

        optimizer_config = self.config.optimizer
        self.optimizer = optim.Adam(self.nflow.parameters(),
                                    lr=optimizer_config.lr,
                                    betas=optimizer_config.betas)

    def train_epoch(self, epoch_idx):

        feat_dataiter = iter(self.feat_loader)

        loss_avg = 0.0
        for train_step in tqdm(range(1,
                                     len(feat_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            feats = next(feat_dataiter)['data'].cuda()
            self.nflow.zero_grad()
            loss = self.nflow.forward_kld(feats.flatten(1))
            loss.backward()
            self.optimizer.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced
