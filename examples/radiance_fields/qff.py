
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .sh import spherical_harmonics
from .ngp import trunc_exp

class QFF(nn.Module):
    def __init__(self, x_dim, min_log2_freq=0, max_log2_freq=5, num_freqs=6, num_quants=80, num_feats=4, num_corrs=8, std=0.0001):
        super().__init__()
        self.x_dim = x_dim
        self.min_log2_freq = min_log2_freq
        self.max_log2_freq = max_log2_freq
        self.num_freqs = num_freqs
        self.num_quants = num_quants
        self.num_feats = num_feats
        self.num_corrs = num_corrs
        self.std = std
        self.qff_type = -1

        freqs = 2.0 ** torch.linspace(min_log2_freq, max_log2_freq, num_freqs)
        self.freqs = nn.Parameter(freqs, False)

    def __repr__(self):
        NF = self.num_freqs
        C = self.num_feats
        Q = self.num_quants
        R = self.num_corrs
        M = self.min_log2_freq
        X = self.max_log2_freq
        return f"QFF{self.qff_type}(m{M}x{X}f{NF}q{Q}c{C}r{R})[3 -> {self.latent_dim}]"

    @property
    def latent_dim(self) -> int:
        return self.num_freqs * 2 * self.num_feats

    # def pos_encode(self, points):
    #     # faster encoding for volume
    #     N = points.shape[0]
    #     freq_points = points.view(1, N, 3) * self.freqs.to(points).view(-1, 1, 1)
    #     return torch.stack((freq_points.sin(), freq_points.cos()), 1).view(-1, N, 1, 1, 3)

    def pos_encode(self, points):
        '''
        points: Nx3
        return: (Fx2x3)xN
        '''
        N = points.shape[0]
        # 1x3xN * Fx1x1 => Fx3xN
        freq_points = points.T.view(1, 3, N) * self.freqs.view(-1, 1, 1)
        # Fx2x3xN => (F23)xN
        return torch.stack((freq_points.sin(), freq_points.cos()), 1).view(-1, N)

    def encode(self, pos_enc):
        raise NotImplementedError

    def forward(self, points):
        """
        must return features given points (and optional dirs)
        """
        if any([s == 0 for s in points.shape]):
            return torch.zeros((0, self.latent_dim)).to(points)
        enc = self.pos_encode(points)
        return self.encode(enc)

class QFF1(QFF):
    def __init__(self, x_dim, min_log2_freq=0, max_log2_freq=5, num_freqs=6, num_quants=80, num_feats=4, num_corrs=8, std=0.0001):
        # std must be scaled lower since we compute 'correlation of input'
        super().__init__(x_dim, min_log2_freq, max_log2_freq, num_freqs, num_quants, num_feats, num_corrs, std ** (1.0 / x_dim))
        self.qff_type = 1

        f = torch.randn((num_freqs * 2 * x_dim, num_feats*num_corrs, num_quants, 1)) * self.std
        self.qff_vector = nn.Parameter(f, True)

    def encode_by_vol(self, encs):
        # (Fx2x3)xN
        N = encs.shape[1]
        NF = self.num_freqs
        C = self.num_feats
        Q = self.num_quants
        R = self.num_corrs

        qvec = self.qff_vector.view(NF*2, 3, C, R, Q)
        qx = qvec[:,  0].view(NF*2, C, R, 1, 1, Q)
        qy = qvec[:,  1].view(NF*2, C, R, 1, Q, 1)
        qz = qvec[:,  2].view(NF*2, C, R, Q, 1, 1)
        qv = (qx * qy * qz).view(NF*2, C, R, Q, Q, Q).sum(2)

        # (Fx2x3)xCx1xN
        return F.grid_sample(qv, encs, mode='bilinear', align_corners=True).view(-1, N).T

    def encode(self, encs):
        # (Fx2x3)xN
        N = encs.shape[-1]
        NF = self.num_freqs
        C = self.num_feats
        Q = self.num_quants
        R = self.num_corrs

        encs = encs.view(-1, 1, N, 1)
        w = torch.zeros_like(encs)
        grid = torch.cat((w, encs), -1)
        grids = grid.view(NF, 2, 3, -1, 1, 2)

        # (Fx2x3)xCx1xN
        qff_v = self.qff_vector.view(NF, 2, 3, C*R, Q, 1)
        fs = []
        for freq in range(NF):
            for sc in range(2):
                f = 1
                for axis in range(3):
                    # 1xCxQx1
                    # 1xNx1x2
                    af =  F.grid_sample(qff_v[freq, sc, axis][None], grids[freq, sc, axis][None], mode='bilinear', align_corners=True)
                    f *= af.view(C, R, N)
                f = f.sum(-2)
                fs.append(f)
        fs = torch.stack(fs)
        return fs.view(-1, N).T

    def encode_2(self, encs):
        # (Fx2x3)xN
        N = encs.shape[-1]
        NF = self.num_freqs
        C = self.num_feats
        Q = self.num_quants
        R = self.num_corrs

        encs = encs.view(-1, 1, N, 1)
        w = torch.zeros_like(encs)
        grid = torch.cat((w, encs), -1)
        grids = grid.view(NF*2,3, -1, 1, 2)

        # (Fx2x3)xCx1xN
        qff_v = self.qff_vector.view(NF*2 ,3, C*R, Q, 1)
        f = 1
        for axis in range(3):
            # 1xCxQx1
            # 1xNx1x2
            f *= F.grid_sample(qff_v[:, axis], grids[:, axis], mode='bilinear', align_corners=True).view(NF*2, C, R, N)
        f = f.sum(-2)
        return f.view(-1, N).T

    def volume(self):
        F = self.num_freqs
        C = self.num_feats
        R = self.num_corrs
        Q = self.num_quants
        qvec = self.qff_vector.view(F*2, 3, C, R, Q)
        qx = qvec[:,  0].view(F*2, C, R, 1, 1, Q)
        qy = qvec[:,  1].view(F*2, C, R, 1, Q, 1)
        qz = qvec[:,  2].view(F*2, C, R, Q, 1, 1)
        qv = (qx * qy * qz).view(F*2, C, R, Q, Q, Q).sum(2)
        return qv

    def get_buffer(self):
        F = self.num_freqs
        C = self.num_feats
        R = self.num_corrs
        Q = self.num_quants
        # 3xF*2*Q*R*C
        return self.qff_vector.data.view(F, 2, 3, C, R, Q).permute(2, 0, 1, 5, 4, 3).cpu().numpy()



class QFF2(QFF):
    def __init__(self, x_dim, min_log2_freq=0, max_log2_freq=5, num_freqs=6, num_quants=80, num_feats=4, num_corrs=8, std=0.0001):
        super().__init__(x_dim, min_log2_freq, max_log2_freq, num_freqs, num_quants, num_feats, num_corrs, std)
        self.qff_type = 2

        v= torch.randn((num_freqs * 2 * x_dim, num_feats * num_corrs, num_quants, 1)) * std
        m = torch.randn((num_freqs * 2 * x_dim, num_feats * num_corrs, num_quants, num_quants)) * std

        self.qff_vector = nn.Parameter(v, True)
        self.qff_plane = nn.Parameter(m, True)

    def mat_inds(self, encs):
        N = encs.shape[-1]
        # 0:xs 1:ys 2:zs, 3:xc 4:yc 5:zc
        encs = encs.view(-1, 6, N)
        # Fx1xN
        xs, ys, zs, xc, yc, zc = encs.view(-1, 6, N).chunk(6, 1)

        return torch.cat([
            torch.stack([ys, zs], -1),
            torch.stack([xs, zs], -1),
            torch.stack([xs, ys], -1),
            torch.stack([yc, zc], -1),
            torch.stack([xc, zc], -1),
            torch.stack([xc, yc], -1)
        ], 1).view(-1, 1, N, 2)

    def encode(self, encs):
        # (Fx2x3)xN
        N = encs.shape[-1]
        NF = self.num_freqs
        C = self.num_feats
        Q = self.num_quants
        R = self.num_corrs


        mat_encs = self.mat_inds(encs)
        encs = encs.view(-1, 1, N, 1)
        w = torch.zeros_like(encs)
        grid = torch.cat((w, encs), -1)
        vec_f = F.grid_sample(self.qff_vector, grid, mode='bilinear', align_corners=True).view(NF, 2, 3, C, R, N)
        mat_f = F.grid_sample(self.qff_plane, mat_encs, mode='bilinear', align_corners=True).view(NF, 2, 3, C, R, N)

        features = (vec_f * mat_f)
        xf, yf, zf = features.chunk(3, 2)
        return (xf + yf +zf).sum(-2).view(-1, N).T

    def volume(self):
        NF = self.num_freqs
        C = self.num_feats
        Q = self.num_quants
        R = self.num_corrs
        vx, vy, vz = self.qff_vector.view(NF*2, 3, C, R, Q).chunk(3, 1)
        pyz, pxz, pxy = self.qff_plane.view(NF*2, 3, C, R, Q, Q).chunk(3, 1)

        f = vx.view(NF*2, C, R, 1, 1, Q) * pyz.view(NF*2, C, R, Q, Q, 1) + \
            vy.view(NF*2, C, R, 1, Q, 1) * pxz.view(NF*2, C, R, Q, 1, Q) + \
            vz.view(NF*2, C, R, Q, 1, 1) * pxy.view(NF*2, C, R, 1, Q, Q)
        return f.sum(2).view(NF*2, C, Q, Q, Q)

    def get_buffer(self):
        F = self.num_freqs
        C = self.num_feats
        Q = self.num_quants
        # Fx2xQxQxQxC
        return self.qff_volume.data.view(F, 2, C, Q, Q, Q).permute(0, 1, 3, 4, 5, 2).cpu().numpy()



class QFF3(QFF):
    def __init__(self, x_dim, min_log2_freq=0, max_log2_freq=5, num_freqs=6, num_quants=80, num_feats=4, num_corrs=1, std=0.0001):
        super().__init__(x_dim, min_log2_freq, max_log2_freq, num_freqs, num_quants, num_feats, 1, std)
        self.qff_type = 3
        f = torch.randn((num_freqs * 2, num_feats, num_quants, num_quants, num_quants)) * std
        self.qff_volume = nn.Parameter(f, True)

    def pos_encode(self, points):
        # faster encoding for volume
        N = points.shape[0]
        freq_points = points.view(1, N, 3) * self.freqs.to(points).view(-1, 1, 1)
        return torch.stack((freq_points.sin(), freq_points.cos()), 1).view(-1, N, 1, 1, 3)

    def encode(self, encs):
        NF = self.num_freqs
        C = self.num_feats
        features = F.grid_sample(self.qff_volume, encs, mode='bilinear', align_corners=True).view(NF*2*C, -1)
        return features.T

    def volume(self):
        return self.qff_volume

    def get_buffer(self):
        # f = (torch.randn((num_freqs * 2 * x_dim, num_feats*num_corrs, num_quants, 1)) * self.std) ** (1 / 3)
        F = self.num_freqs
        C = self.num_feats
        Q = self.num_quants
        # Fx2xQxQxQxC
        return self.qff_volume.data.view(F, 2, C, Q, Q, Q).permute(0, 1, 3, 4, 5, 2).cpu().numpy()


class QFFRadianceField(torch.nn.Module):
    """QFF Radiance Field"""

    def __init__(
        self,
        num_dim: int = 3,
        density_activation= lambda x: trunc_exp(x),
        qff_type=1, 
        num_quants=80, 
        num_features=4,  
        min_log2_freq=1,
        max_log2_freq=6,
        num_freqs=4,
        num_corrs=16,
    ) -> None:
        super().__init__()
        self.num_dim = num_dim
        self.density_activation = density_activation

        self.qff_type = qff_type
        self.num_quants = num_quants
        self.num_features = num_features
        self.min_log2_freq = min_log2_freq
        self.max_log2_freq = max_log2_freq
        self.num_freqs = num_freqs
        args = {
            'x_dim': num_dim,
            'min_log2_freq': min_log2_freq,
            'max_log2_freq': max_log2_freq,
            'num_freqs': num_freqs,
            'num_quants': num_quants,
            'num_feats': num_features,
            'num_corrs': num_corrs,
        }
        # for QFF3, num_corrs will always be 1
        self.encoder = eval(f"QFF{qff_type}")(**args)
        self.mlp =  nn.Linear(self.encoder.latent_dim, 8, bias=False)

    def __repr__(self):
        return f'{self.encoder} => GeomMLP[{self.encoder.latent_dim} -> 64 -> Density[1] + 15] => ColorMLP[15 + SH[Dir[3] -> 16]-> 64 -> 64 -> RGB]'

    
    def get_qff_buffer(self):
        F = self.num_freqs
        qff_buffer = self.encoder.get_buffer().astype(np.float16).tobytes()
        qff_dv = self.qff_dv()
        # sum of (average all pixels) over all frequencies
        qff_mean = qff_dv.view(F*2, -1).mean(1).sum(0).item()
        return qff_buffer, qff_mean

    def qff_v(self):
        return self.encoder.volume()

    def qff_dv(self):
        F = self.encoder.num_freqs
        Q = self.encoder.num_quants
        C = self.encoder.num_feats
        qff_v = self.encoder.volume()
        # sum of channels of alpha dxyzs per frequencies x 2
        # 8 values represent: rgba, dxyz. Only take alpha
        qff_dv = (qff_v.view(F*2, C, 1, Q, Q, Q) * self.mlp.weight.T.view(F*2, C, 8, 1, 1, 1))[:, :, 3].sum(1)

        return qff_dv

    def query_density(self, x, return_feat: bool = False):
        enc = self.encoder(x)
        f = self.mlp(enc)
        # return self.density_activation(f[..., -1:])
        s = self.density_activation(f[..., 3])
        return s[..., None]

    def forward(self, positions, directions=None):
        enc = self.encoder(positions)
        f = self.mlp(enc)
        # rad = f[..., :-1].view(*f.shape[:-1], 3, 4)

        # s = self.density_activation(f[..., -4:])

        # Bx4
        dxyz = torch.cat([directions, torch.ones_like(directions[..., :1])], 1)
        rgba = f[..., :4]
        df = (f[..., 4:] * dxyz).sum(-1, True)

        # rgb = (rgba[..., :3] + df).sigmoid()
        rgb = trunc_exp((rgba[..., :3] + df).clamp_max(11))
        a = trunc_exp(rgba[..., 3:].clamp_max(11))

        return rgb, a