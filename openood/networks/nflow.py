import normflows as nf
import torch


class ClampedMLP(nf.nets.MLP):
    def __init__(self, clamp_value=None, **kwargs):
        super().__init__(**kwargs)
        self.clamp_value = clamp_value

    def forward(self, x):
        x = super().forward(x)
        if self.clamp_value is None:
            return x
        x = torch.clamp(x, min=-self.clamp_value, max=self.clamp_value)
        return x


def get_normalizing_flow(network_config):
    latent_size = network_config.latent_size
    hidden_size = network_config.hidden_size
    if hidden_size is None:
        hidden_size = latent_size * 2
    n_flows = network_config.n_flows
    clamp_value = network_config.clamp_value
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
    flows = []
    for i in range(n_flows):
        s = ClampedMLP(clamp_value=clamp_value,
                       layers=[latent_size, hidden_size, latent_size],
                       init_zeros=True)
        t = ClampedMLP(clamp_value=clamp_value,
                       layers=[latent_size, hidden_size, latent_size],
                       init_zeros=True)
        if i % 2 == 0:
            flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        else:
            flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        flows += [nf.flows.ActNorm(latent_size)]

    q0 = nf.distributions.DiagGaussian(latent_size)
    # Construct flow model
    nfm = nf.NormalizingFlow(q0=q0, flows=flows)
    return nfm
