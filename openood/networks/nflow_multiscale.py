import torch
from torch import nn

from openood.networks.nflow import get_normalizing_flow
from openood.utils import Config


class MultiScaleNormalizingFlow(nn.Module):
    def __init__(self, nflows, scale_sizes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nflows = nn.ModuleList(nflows)
        self.scale_sizes = scale_sizes

    def _apply_to_scales(self, x, func):
        assert x.size(1) == sum(self.scale_sizes), \
            f'Input size {x.size(1)} does not match sum of scale sizes ' \
            f'{sum(self.scale_sizes)}'
        results = []
        start = 0
        for size, flow in zip(self.scale_sizes, self.nflows):
            end = start + size
            results.append(func(flow, x[:, start:end]))
            start = end
        return results

    def forward_kld(self, x):
        klds = self._apply_to_scales(
            x, lambda flow, feats: flow.forward_kld(feats))
        return torch.mean(torch.stack(klds))

    def inverse(self, x):
        zs = self._apply_to_scales(x, lambda flow, feats: flow.inverse(feats))
        return torch.cat(zs, dim=1)

    def log_prob(self, x):
        log_probs = self._apply_to_scales(
            x, lambda flow, feats: flow.log_prob(feats))
        return torch.mean(torch.stack(log_probs), dim=0)


def get_multiscale_normalizing_flow(network_config):
    latent_size = list(map(int, network_config.latent_size))
    hidden_size = list(map(int, network_config.hidden_size))
    n_flows = list(map(int, network_config.n_flows))
    nflows = []
    for l_sz, h_sz, n_fs in zip(latent_size, hidden_size, n_flows):
        net_config = Config(network_config)
        net_config.latent_size = l_sz
        net_config.hidden_size = h_sz
        net_config.n_flows = n_fs
        nflow = get_normalizing_flow(net_config)
        nflows.append(nflow)

    ms_nfm = MultiScaleNormalizingFlow(nflows, scale_sizes=latent_size)
    return ms_nfm
