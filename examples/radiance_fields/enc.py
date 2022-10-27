import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class FreqEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, log2_res=7, num_feats=8, std=0.001, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.num_feats = num_feats
        self.use_identity = use_identity
        scales = 2.0 ** torch.linspace(min_deg, max_deg, max_deg - min_deg)
        #  "scales", torch.linspace(2.0 ** min_deg, 2.0**max_deg, max_deg - min_deg)
        self.register_buffer(
             "scales", scales
        )
        res = 2 ** log2_res
        num_scales = (self.max_deg - self.min_deg)
        features = torch.randn((num_scales * 2 * 3, num_feats, res, 1)) * std
        self.features = nn.Parameter(features, True)

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + (self.max_deg - self.min_deg) * 2 * self.num_feats
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
        # Nx1x3 * Fx1 => NxFx3 => NxF3
        num_scales = (self.max_deg - self.min_deg)
        num_feats = num_scales * self.x_dim * 2
        num_channels = self.features.shape[1]

        xb = torch.reshape((x[Ellipsis, None, :] * self.scales[:, None]), list(x.shape[:-1]) + [num_scales, self.x_dim])

        # NxFx2x3
        # features = torch.randn((num_encodings * 2 * 3, num_feats, res, 1)) * std

        # NxFx2x3
        latent = torch.sin(torch.stack([xb, xb + 0.5 * math.pi], dim=-2))
        grid_latent = latent.view(-1, num_feats).T.view(num_feats, 1, -1, 1)
        zs = torch.zeros_like(grid_latent)
        grid = torch.cat((zs, grid_latent), -1)
        fs = F.grid_sample(self.features, grid, mode='bilinear', align_corners=True).view(num_scales, 2, self.x_dim, num_channels, -1)
        # NxCxFx2x3
        latent = (fs.permute(4, 3, 0, 1, 2) + latent.view(-1, 1, num_scales, 2, self.x_dim)).reshape(*x.shape[:-1], -1)

        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent

class MultiFreqEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, res_scale=4, num_feats=8, std=0.001, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.num_feats = num_feats
        self.use_identity = use_identity
        # scales = 2.0 ** torch.linspace(min_deg, max_deg, max_deg - min_deg)
        scales = torch.tensor([2**i for i in range(min_deg, max_deg)])
        int_scales = scales.long()
        features = []
        for i, scale in enumerate(int_scales):
            feature = torch.randn((2 * x_dim, num_feats, scale * res_scale, 1)) * std
            features.append(nn.Parameter(feature, True))
        self.register_buffer(
             "scales", scales
        )
        self.features = nn.ParameterList(features)

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + (self.max_deg - self.min_deg) * 2 * self.num_feats
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
        if any([s == 0 for s in x.shape]):
            return torch.zeros((0, self.latent_dim)).to(x)

        # Nx1x3 * Fx1 => NxFx3 => NxF3
        num_scales = (self.max_deg - self.min_deg)

        xb = torch.reshape((x[Ellipsis, None, :] * self.scales[:, None]), list(x.shape[:-1]) + [num_scales, self.x_dim])
        latent = torch.sin(torch.stack([xb, xb + 0.5 * math.pi], dim=-2))
        # latent = torch.sin(torch.stack([xb, xb + 0.5 * math.pi], dim=-2)).unsqueeze(1).repeat(1, self.num_feats, 1, 1, 1).view(*x.shape[:-1], -1)
        # Fx23xN
        grid_latent = latent.view(-1, num_scales, 2 * self.x_dim).permute(1, 2, 0)
        zs = torch.zeros_like(grid_latent)
        # Fx23xNx2
        grid = torch.stack((zs, grid_latent), -1).unsqueeze(2)

        latents = []
        for i, scale in enumerate(self.scales):
            # 23xCx1xN + 23x1x1xN
            num_channels = self.features[i].shape[1]
            fs = F.grid_sample(self.features[i], grid[i], mode='bilinear', align_corners=True).view(2, self.x_dim, num_channels, -1)
            latents.append(fs.permute(3, 2, 0, 1))

        # Fx2x3xCxN
        latent = (torch.stack(latents, 2) + latent.view(-1, 1, num_scales, 2, self.x_dim)).reshape(*x.shape[:-1], -1)

        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent

class MultiFreqEncoder2D(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, res_scale=4, num_feats=8, std=0.001, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.num_feats = num_feats
        self.use_identity = use_identity
        scales = 2.0 ** torch.linspace(min_deg, max_deg, max_deg - min_deg)
        # scales = torch.tensor([2**i for i in range(min_deg, max_deg)])
        int_scales = scales.long()
        features_2d = []
        features = []
        for i, scale in enumerate(int_scales):
            feature = torch.randn((2 * x_dim, num_feats, scale * res_scale, scale * res_scale)) * std
            # s2d = int(math.log2(scale * res_scale) * res_scale)
            feature_2d = torch.randn((2 * x_dim, num_feats, scale * res_scale, scale * res_scale)) * std
            features.append(nn.Parameter(feature, True))
            features_2d.append(nn.Parameter(feature_2d, True))
        self.register_buffer(
             "scales", scales
        )
        self.features = nn.ParameterList(features)
        self.features_2d = nn.ParameterList(features_2d)

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + (self.max_deg - self.min_deg) * 2 * self.num_feats
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
        # Nx1x3 * Fx1 => NxFx3 => NxF3
        num_scales = (self.max_deg - self.min_deg)

        xb = torch.reshape((x[Ellipsis, None, :] * self.scales[:, None]), list(x.shape[:-1]) + [num_scales, self.x_dim])
        latent = torch.sin(torch.stack([xb, xb + 0.5 * math.pi], dim=-2)).view(-1, num_scales, 2, self.x_dim)

        # create grid for 1d
        grid_latent = latent.view(-1, num_scales, 2 * self.x_dim).permute(1, 2, 0)
        zs = torch.zeros_like(grid_latent)
        grid = torch.stack((zs, grid_latent), -1).unsqueeze(2)

        # create grid for 2d
        # NxFx2x3 => 6[NxFx2] => NxFx6x2 => Fx6xNx2
        grid_2d = torch.stack([
            latent[..., 0, [1, 2]],
            latent[..., 0, [2, 0]],
            latent[..., 0, [0, 1]],
            latent[..., 1, [1, 2]],
            latent[..., 1, [2, 0]],
            latent[..., 1, [0, 1]],
        ], 2).permute(1, 2, 0, 3).unsqueeze(2).contiguous()

        latents = []
        for i, scale in enumerate(self.scales):
            # 23xCx1xN + 23x1x1xN
            num_channels = self.features[i].shape[1]
            fs = F.grid_sample(self.features[i], grid[i], mode='bilinear', align_corners=True).view(2, self.x_dim, num_channels, -1)
            fs_2d = F.grid_sample(self.features_2d[i], grid_2d[i], mode='bilinear', align_corners=True).view(2, self.x_dim, num_channels, -1)
            latents.append((fs * fs_2d).permute(3, 2, 0, 1))

        # Fx2x3xCxN
        latent = (torch.stack(latents, 2) + latent.view(-1, 1, num_scales, 2, self.x_dim)).reshape(*x.shape[:-1], -1)

        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent