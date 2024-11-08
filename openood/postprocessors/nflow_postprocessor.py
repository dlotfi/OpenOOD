from typing import Any

import torch

from .base_postprocessor import BasePostprocessor


class NormalizingFlowPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(NormalizingFlowPostprocessor, self).__init__(config)

    @torch.no_grad()
    def postprocess(self, net, data: Any):
        # images input
        if data.shape[-1] > 1 and data.shape[1] in {1, 3}:
            data = torch.Tensor(data)  # fix the issue when data is MetaTensor
            output, feats = net['backbone'](data, return_feature=True)
            score = torch.softmax(output, dim=1)
            _, pred = torch.max(score, dim=1)
            if 'feat_agg' in net:
                feats = net['feat_agg'](feats)
            log_prob = net['nflow'].log_prob(feats)
            log_prob = log_prob.view(-1, 1)
            conf = log_prob.reshape(-1).detach().cpu()
        # feature input
        elif data.shape[-1] == 1 and data.shape[-2] == 1:
            feats = data.flatten(1)
            if 'feat_agg' in net:
                feats = net['feat_agg'](feats)
            log_prob = net['nflow'].log_prob(feats)
            log_prob = log_prob.view(-1, 1)
            conf = log_prob.reshape(-1).detach().cpu()
            pred = torch.ones_like(conf)  # dummy predictions
        else:
            raise ValueError('Unsupported input type!')

        return pred, conf
