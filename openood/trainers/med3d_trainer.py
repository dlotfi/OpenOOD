from functools import partial

import torch
import torch.nn.functional as F
import torch.optim as optim

from .base_trainer import BaseTrainer


class Med3DTrainer(BaseTrainer):
    def get_optimizer(self):
        optimizer_config = self.config.optimizer
        if optimizer_config.name.lower() == 'sgd':
            return optim.SGD(
                self.net.parameters(),
                lr=optimizer_config.lr,
                momentum=optimizer_config.momentum,
                weight_decay=optimizer_config.weight_decay,
                nesterov=True,
            )
        elif optimizer_config.name.lower() == 'adam':
            return optim.Adam(self.net.parameters(),
                              lr=optimizer_config.lr,
                              betas=optimizer_config.betas,
                              weight_decay=optimizer_config.weight_decay)
        else:
            raise ValueError('Optimizer not supported')

    def get_scheduler(self):
        scheduler_config = self.config.scheduler
        total_steps = self.config.optimizer.num_epochs * len(self.train_loader)
        if scheduler_config is None:
            return type('DummyScheduler', (), {'step': lambda *a: None})()
        elif scheduler_config.name.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                        T_max=total_steps,
                                                        eta_min=1e-6)
        elif scheduler_config.name.lower() == 'poly':
            return optim.lr_scheduler.PolynomialLR(
                self.optimizer,
                total_iters=total_steps,
                power=scheduler_config.power,
            )
        elif scheduler_config.name.lower() == 'multi-step':
            return optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=scheduler_config.milestones,
                gamma=scheduler_config.gamma)
        else:
            raise ValueError('Scheduler not supported')

    def get_loss_fn(self):
        loss_config = self.config.loss
        if loss_config is None:
            return super().get_loss_fn()
        elif loss_config.name.lower() == 'weighted_ce':
            weight = torch.Tensor(loss_config.weight).cuda()
            return partial(F.cross_entropy, weight=weight)
        else:
            raise ValueError('Loss function not supported')
