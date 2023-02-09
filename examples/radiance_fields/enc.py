import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class QFF(nn.Module):
    def __init__(self, input_width, min_log2_freq, max_log2_freq, num_freqs=16, quant_size=64, num_feats=1, std=0.0001, use_identity: bool = True):
        super().__init__()
        self.input_width =input_width 
        self.num_freqs = num_freqs

        self.num_freqs = num_freqs
        self.use_identity = use_identity
        self.num_feats = num_feats

        # freqs = torch.tensor([2**i for i in linspace(min_log2_freq, max_log2_freqs, num_freqs)])
        freqs = 2.0 ** torch.linspace(min_log2_freq, max_log2_freq, num_freqs)
        self.freqs = nn.Parameter(freqs, False)

        cv = torch.randn((num_freqs * 8, num_feats, quant_size, quant_size, quant_size)) * std
        self.cv = nn.Parameter(cv, True)

    
    @property
    def latent_dim(self) -> int:
        return self.output_width


    @property
    def output_width(self) -> int:
        return (int(self.use_identity) * self.input_width) + self.num_freqs * 8 * self.num_feats

    def pos_encode(self, points):
        '''
        Bx3 => FxBx8x3
        '''
        N = points.shape[0]
        # FN3
        F = len(self.freqs)
        freq_points = points.view(1, N, -1) * self.freqs.to(points).view(-1, 1, 1)
        sins = freq_points.sin()
        coss = freq_points.cos()
        # FxNx6 : (sx, sy, sz, cx, cy, cz)
        scs = torch.cat([sins, coss], -1)
        indices = [
            0, 1, 2, # sss
            0, 1, 5, # ssc
            0, 4, 2, # scs
            0, 4, 5, # scc
            3, 1, 2, # css
            3, 1, 5, # csc
            3, 4, 2, # ccs
            3, 4, 5, # ccc
        ]
        return scs[..., indices].view(F, N, 8, 3).permute(0, 2, 1, 3).reshape(F*8, 1, 1, N, 3)

    def forward(self, points):
        """
        must return features given points (and optional dirs)
        """
        if any([s == 0 for s in points.shape]):
            return torch.zeros((0, self.latent_dim)).to(points)

        enc = self.pos_encode(points)
        # F8xCx1x1xN => F8xCxN
        F = len(self.freqs)
        C = self.num_feats
        N = len(points)

        # Nx(FC8)
        features = nn.functional.grid_sample(self.cv, enc, mode='bilinear', align_corners=False)[:, :, 0, 0].view(F* 8* C, N).T * 10
        if self.use_identity:
            features = torch.cat((points, features), 1)

        return features

class QFFLite(nn.Module):

    def __init__(self, input_width, min_log2_freq, max_log2_freq, num_freqs=16, quant_size=64, num_feats=1, std=0.0001, use_identity: bool = True):
        super().__init__()
        self.input_width =input_width 
        self.num_freqs = num_freqs

        self.num_freqs = num_freqs
        self.use_identity = use_identity
        self.num_feats = num_feats

        # freqs = torch.tensor([2**i for i in linspace(min_log2_freq, max_log2_freqs, num_freqs)])
        freqs = 2.0 ** torch.linspace(min_log2_freq, max_log2_freq, num_freqs)
        self.freqs = nn.Parameter(freqs, False)

        cv = torch.randn((num_freqs * 2, num_feats, quant_size, quant_size, quant_size)) * std
        self.cv = nn.Parameter(cv, True)

    
    @property
    def latent_dim(self) -> int:
        return self.output_width


    @property
    def output_width(self) -> int:
        return (int(self.use_identity) * self.input_width) + self.num_freqs * 2 * self.num_feats

    def pos_encode(self, points):
        '''
        Bx3 => FxBx8x3
        '''
        N = points.shape[0]
        # FN3
        F = len(self.freqs)
        freq_points = points.view(1, N, -1) * self.freqs.to(points).view(-1, 1, 1)
        sins = freq_points.sin()
        coss = freq_points.cos()
        # FxNx6 : (sx, sy, sz, cx, cy, cz)
        scs = torch.cat([sins, coss], -1)
        indices = [
            0, 1, 2, # sss
            3, 4, 5, # sss
        ]
        return scs[..., indices].view(F, N, 2, 3).permute(0, 2, 1, 3).reshape(F*2, 1, 1, N, 3)

    def forward(self, points):
        """
        must return features given points (and optional dirs)
        """
        if any([s == 0 for s in points.shape]):
            return torch.zeros((0, self.latent_dim)).to(points)

        enc = self.pos_encode(points)
        # F8xCx1x1xN => F8xCxN
        F = len(self.freqs)
        C = self.num_feats
        N = len(points)

        # Nx(FC8)
        features = nn.functional.grid_sample(self.cv, enc, mode='bilinear', align_corners=False)[:, :, 0, 0].view(F* 2* C, N).T 
        if self.use_identity:
            features = torch.cat((points, features), 1)

        return features
