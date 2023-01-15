import math
import torch
import torch.nn as nn

class SinusoidalEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer(
            "scales", torch.tensor([2**i for i in range(min_deg, max_deg)])
        )

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + (self.max_deg - self.min_deg) * 2
        ) * self.x_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim],
        )
        latent = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent

class MultiFourierEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, num_features, min_deg, max_deg, use_identity: bool = True, use_log_sampling=False):
        super().__init__()
        # compute lognormal mean, std from  render_dist

        # sample
        self.x_dim = x_dim
        self.num_features = num_features
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.use_log_sampling = use_log_sampling

        # create randomized fourier mappings
        min_s = 2 ** min_deg
        max_s = 2 ** max_deg

        if self.use_log_sampling:
            scales = torch.rand(num_features * 3) * (max_s - min_s) + min_s
        else:
            scales = 2 ** (torch.rand(num_features * 3) * (max_deg- min_deg) + min_deg)
        phases = torch.rand(num_features * 3) * math.pi * 2
        self.scales = nn.Parameter(scales.view(-1, 3), False)
        self.phases = nn.Parameter(phases.view(-1, 3), False)

    @property
    def latent_dim(self) -> int:
        return int(self.use_identity) * self.x_dim + self.num_features
         

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if len(x) == 0:
            return torch.zeros((0, self.latent_dim)).to(x)
        xs = x[..., None, :] * self.scales[None] + self.phases[None]
        latent = xs.sin().prod(-1).view(*x.shape[:-1], -1)
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent

class FourierEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, num_features, min_deg, max_deg, use_identity: bool = True):
        super().__init__()
        # compute lognormal mean, std from  render_dist

        # sample
        self.x_dim = x_dim
        self.num_features = num_features
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity

        # create randomized fourier mappings
        min_s = 2 ** min_deg
        max_s = 2 ** max_deg

        if self.use_log_sampling:
            scales = torch.rand(num_features * 3) * (max_s - min_s) + max_s
        else:
            scales = 2 ** (torch.rand(num_features * 3) * (max_deg- min_deg) + min_deg)
        phases = torch.rand(num_features * 3) * math.pi * 2
        self.scales = nn.Parameter(scales.view(-1, 3), False)
        self.phases = nn.Parameter(phases.view(-1, 3), False)

    @property
    def latent_dim(self) -> int:
        return (int(self.use_identity) + self.num_features)* self.x_dim
         

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        xs = x[..., None, :] * self.scales[None] + self.phases[None]
        latent = xs.sin().view(*x.shape[:-1], -1)
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent