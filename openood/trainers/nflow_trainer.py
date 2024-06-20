import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import openood.utils.comm as comm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NormalizingFlowTrainer:
    def __init__(self, net, feat_loader, config) -> None:

        # manualSeed = 999
        # print('Random Seed: ', manualSeed)
        # random.seed(manualSeed)
        # torch.manual_seed(manualSeed)

        self.config = config
        self.net = net
        self.nflow = net['nflow']
        self.nflow.apply(weights_init)
        self.feat_loader = feat_loader

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
