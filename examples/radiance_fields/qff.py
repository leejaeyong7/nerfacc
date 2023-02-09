"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import Callable, List, Union

import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from .enc import QFF,QFFLite

try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


class QFFRadianceField(torch.nn.Module):
    """Instance-NGP radiance Field"""

    def __init__(
        self,
        num_freqs=16,
        min_log2_freq=0,
        max_log2_freq=5,
        num_feature_per_freq=4,
        quant_level=64,
        geo_feat_dim: int = 15,
    ) -> None:
        super().__init__()

        self.geo_feat_dim = geo_feat_dim

        self.posi_encoder = QFFLite(
            3, min_log2_freq, max_log2_freq,
            num_freqs, quant_level, num_feature_per_freq,
            0.00001, True
        )

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3,
                        "otype": "SphericalHarmonics",
                        "degree": 4,
                    },
                    # {"otype": "Identity", "n_bins": 4, "degree": 4},
                ],
            },
        )

        self.proj = tcnn.Network(
            n_input_dims=self.posi_encoder.latent_dim,
            n_output_dims=1+self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )

        if self.geo_feat_dim > 0:
            self.mlp_head = tcnn.Network(
                n_input_dims=(self.direction_encoding.n_output_dims + self.geo_feat_dim ),
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )

    def query_density(self, x, return_feat: bool = False):
        x = (
            self.proj(self.posi_encoder(x.view(-1, 3)))
            .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(x)
        )

        density_before_activation, base_mlp_out = torch.split(
            x, [1, self.geo_feat_dim], dim=-1
        )

        density = trunc_exp(density_before_activation - 1)

        if return_feat:
            return density, base_mlp_out
        else:
            return density

    def _query_rgb(self, dir, embedding):
        # tcnn requires directions in the range [0, 1]
        dir = (dir + 1.0) / 2.0
        d = self.direction_encoding(dir.view(-1, dir.shape[-1]))
        h = torch.cat([d, embedding.view(-1, self.geo_feat_dim)], dim=-1)
        rgb = (
            self.mlp_head(h)
            .view(list(embedding.shape[:-1]) + [3])
            .to(embedding)
        )
        return rgb

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor = None,
    ):
        density, embedding = self.query_density(positions, return_feat=True)
        rgb = self._query_rgb(directions, embedding=embedding)
        return rgb, density
