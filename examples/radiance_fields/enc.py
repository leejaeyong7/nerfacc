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

    def __init__(self, x_dim, min_deg, max_deg, num_freqs, log2_res=7, num_feats=8, std=0.001, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.num_feats = num_feats
        self.use_identity = use_identity
        self.num_freqs = num_freqs

        # scales = 2.0 ** torch.linspace(min_deg, max_deg, num_freqs)
        scales = torch.linspace(2.0 ** min_deg, 2.0 ** max_deg, num_freqs)
        # scales = torch.tensor([2**i for i in range(min_deg, max_deg)])
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
            int(self.use_identity) + (self.num_freqs) * 2 * self.num_feats
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
        # num_scales = (self.max_deg - self.min_deg)
        num_scales = self.num_freqs
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


class FreqHashO(nn.Module):
    def __init__(self, x_dim, min_deg=0, max_deg=5, num_freqs=6, log2_res=8, num_feats=8, std=0.1, use_identity=True):
        super().__init__()
        self.x_dim = x_dim
        self.use_identity = use_identity
        self.max_deg = max_deg
        self.min_deg = min_deg
        # num_freqs = max_deg - min_deg
        self.num_freqs = num_freqs
        self.num_feats = num_feats

        # freqs = torch.tensor([2**i for i in range(min_deg, max_deg)])
        scales = torch.linspace(2.0 ** min_deg, 2.0 ** max_deg, num_freqs)
        self.scales = nn.Parameter(scales, False)

        res = 2 ** log2_res
        f = torch.randn((num_freqs * 2 * 3, num_feats, res, 1)) * std
        self.features = nn.Parameter(f, True)

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + (self.num_freqs) * 2 * self.num_feats
        ) * self.x_dim

    def pos_encode(self, points):
        N = points.shape[0]
        freq_points = points.view(N, 1, -1) * self.scales.to(points).view(1, -1, 1)

        # NxFx2x3 => Nx(Fx2x3)
        return torch.stack((freq_points.sin(), freq_points.cos()), -2).view(N, -1).T.contiguous()

    def encoder(self, encs):
        # (Fx2x3)xN
        NF = self.num_freqs
        N = encs.shape[-1]

        # (Fx2x3)xCxRx1
        cv = self.features

        C = cv.shape[1]
        encs = encs.view(-1, 1, N, 1)
        w = torch.zeros_like(encs)
        grid = torch.cat((w, encs), -1)

        # (Fx2x3)xCx1xN
        fs = F.grid_sample(cv, grid, mode='bilinear', align_corners=True).view(NF, -1, 3, C, N)
        fs = fs + encs.view(NF, -1, 3, 1, N)
        return fs.permute(4, 3, 0, 1, 2).reshape(N, -1)

    def forward(self, points):
        """
        must return features given points (and optional dirs)
        """
        if any([s == 0 for s in points.shape]):
            return torch.zeros((0, self.latent_dim)).to(points)
        enc = self.pos_encode(points)
        features = self.encoder(enc)
        if self.use_identity:
            features = torch.cat((points, features), 1)

        return features

class FreqHash(nn.Module):
    def __init__(self, x_dim, min_deg=0, max_deg=5, num_freqs=6, log2_res=8, num_feats=8, std=0.1, use_identity=True):
        super().__init__()
        self.x_dim = x_dim
        self.use_identity = use_identity
        self.max_deg = max_deg
        self.min_deg = min_deg
        # num_freqs = max_deg - min_deg
        self.num_freqs = num_freqs
        self.num_feats = num_feats

        # freqs = torch.tensor([2**i for i in range(min_deg, max_deg)])
        freqs = 2.0 ** torch.linspace(min_deg, max_deg, num_freqs)

        self.freqs = nn.Parameter(freqs, False)
        res = 2 ** log2_res
        f = torch.randn((num_freqs * 2 * 3, num_feats, res, 1)) * std
        self.cv = nn.Parameter(f, True)

        self.features = self.cv

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + (self.num_freqs) * 2 * self.num_feats
        ) * self.x_dim

    def pos_encode(self, points):
        N = points.shape[0]
        freq_points = points.view(N, 1, -1) * self.freqs.to(points).view(1, -1, 1)

        # NxFx2x3 => Nx(Fx2x3)
        return torch.stack((freq_points.sin(), freq_points.cos()), -2).view(N, -1).T.contiguous()

    def encoder(self, encs):
        # (Fx2x3)xN
        NF = self.num_freqs
        N = encs.shape[-1]

        # (Fx2x3)xCxRx1
        cv = self.cv
        cv = self.features

        C = cv.shape[1]
        encs = encs.view(-1, 1, N, 1)
        w = torch.zeros_like(encs)
        grid = torch.cat((w, encs), -1)

        # (Fx2x3)xCx1xN
        fs = F.grid_sample(cv, grid, mode='bilinear', align_corners=True).view(NF, -1, 3, C, N)
        # fs = fs + encs.view(NF, -1, 3, 1, N)
        return fs.permute(4, 3, 0, 1, 2).reshape(N, -1)

    def forward(self, points):
        """
        must return features given points (and optional dirs)
        """
        if any([s == 0 for s in points.shape]):
            return torch.zeros((0, self.latent_dim)).to(points)
        enc = self.pos_encode(points)
        features = self.encoder(enc)
        if self.use_identity:
            features = torch.cat((points, features), 1)

        return features

class FreqVMEncoder(nn.Module):
    def __init__(self, x_dim, min_deg, max_deg, num_freqs=6, log2_res=7, num_feats=8, std=0.001, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        min_log2_freq = min_deg
        max_log2_freq = max_deg
        self.num_freqs = num_freqs

        self.num_freqs = num_freqs
        self.use_identity = use_identity
        self.num_feats = num_feats

        # freqs = torch.tensor([2**i for i in range(min_deg, max_deg)])
        freqs = 2.0 ** torch.linspace(min_log2_freq, max_log2_freq, num_freqs)
        # freqs = torch.linspace(2.0 ** min_log2_freq, 2.0 ** max_log2_freq, num_freqs)
        self.freqs = nn.Parameter(freqs, False)

        res = 2 ** log2_res
        cv = torch.randn((num_freqs * 2 * 3, num_feats, res, 1)) * std
        cm = torch.randn((num_freqs * 2 * 3, num_feats, res, res)) * std

        cv = nn.Parameter(cv, True)
        cm = nn.Parameter(cm, True)

        self.params = nn.ParameterDict({
            'cv': cv,
            'cm': cm,
        })

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + self.num_freqs * 2 * self.num_feats
        ) * self.x_dim

    def pos_encode(self, points):
        N = points.shape[0]
        # NF3
        freq_points = points.view(N, 1, -1) * self.freqs.to(points).view(1, -1, 1)

        # NxFx2x3 => Nx(Fx2x3) => F23xN
        return torch.stack((freq_points.sin(), freq_points.cos()), -2).view(N, -1).T.contiguous()

    def mat_inds(self, encs):
        N = encs.shape[-1]
        # 0:xs 1:ys 2:zs, 3:xc 4:yc 5:zc
        encs = encs.view(-1, 6, N)

        # Fx2xN
        yszs = encs[:, [1, 2]]
        zsxs = encs[:, [2, 0]]
        xsys = encs[:, [0, 1]]
        yczc = encs[:, [4, 5]]
        zcxc = encs[:, [5, 3]]
        xcyc = encs[:, [3, 4]]


        # Fx6x2xN
        return torch.stack([yszs, zsxs, xsys, yczc, zcxc, xcyc], 1).permute(0, 1, 3, 2).view(-1, 1, N, 2).contiguous()

    def encoder(self, encs, mat_grid):
        # (Fx2x3)xN
        N = encs.shape[-1]

        # (Fx2x3)xCxRx1

        # sampling vector
        cv = self.params['cv']
        C = cv.shape[1]
        encs = encs.view(-1, 1, N, 1)
        w = torch.zeros_like(encs)
        grid = torch.cat((w, encs), -1)

        # (Fx2x3)xCx1xN
        vec_f = F.grid_sample(cv, grid, mode='bilinear', align_corners=True).view(-1, 2, 3, C, N)
        cm = self.params['cm']

        # Fx2x3xCxN
        mat_f = F.grid_sample(cm, mat_grid, mode='bilinear', align_corners=True).view(-1, 2, 3, C, N)

        # Fx2x3xCxN
        # basis = self.params['basis']
        fs = (vec_f * mat_f)
        # fs = fs + encs.view(-1, 2, 3, 1, N)

        return fs.permute(4, 3, 0, 1, 2).reshape(N, -1)


    def forward(self, points):
        """
        must return features given points (and optional dirs)
        """
        if any([s == 0 for s in points.shape]):
            return torch.zeros((0, self.latent_dim)).to(points)

        enc = self.pos_encode(points)
        grid = self.mat_inds(enc)
        features = self.encoder(enc, grid)
        if self.use_identity:
            features = torch.cat((points, features), 1)

        return features


