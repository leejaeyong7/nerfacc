"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import random
from typing import Optional

import numpy as np
import torch
from datasets.utils import Rays, namedtuple_map

from nerfacc import OccupancyGrid, ray_marching, accumulate_along_rays, render_weight_from_density
from nerfacc.pack import unpack_info


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def rendering_with_normal(
    # radiance field
    rgb_sigma_fn,
    # ray marching results
    packed_info: torch.Tensor,
    t_starts: torch.Tensor,
    t_ends: torch.Tensor,
    # rendering options
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
    render_bkgd: Optional[torch.Tensor] = None,
):
    n_rays = packed_info.shape[0]
    ray_indices = unpack_info(packed_info)

    # Query sigma and color with gradients
    rgbs, grads, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices.long())
    assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
        rgbs.shape
    )
    assert (
        sigmas.shape == t_starts.shape
    ), "sigmas must have shape of (N, 1)! Got {}".format(sigmas.shape)

    # Rendering: compute weights and ray indices.
    weights = render_weight_from_density(
        packed_info, t_starts, t_ends, sigmas, early_stop_eps, alpha_thre
    )

    # Rendering: accumulate rgbs, opacities, and depths along the rays.
    colors = accumulate_along_rays(
        weights, ray_indices, values=rgbs, n_rays=n_rays
    )
    opacities = accumulate_along_rays(
        weights, ray_indices, values=None, n_rays=n_rays
    )
    normals = accumulate_along_rays(
        weights, ray_indices, values=grads, n_rays=n_rays
    )
    depths = accumulate_along_rays(
        weights,
        ray_indices,
        values=(t_starts + t_ends) / 2.0,
        n_rays=n_rays,
    )

    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)

    return colors, opacities, normals, depths



def render_image_with_normal(
    # scene
    radiance_field: torch.nn.Module,
    occupancy_grid: OccupancyGrid,
    rays: Rays,
    scene_aabb: torch.Tensor,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        ray_indices = ray_indices.long()
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            return radiance_field.query_density(positions, t)
        # check the points for query
        chunk_size = 1024 * 1024
        densities = []
        n = positions.shape[0]
        if n == 0:
            return radiance_field.query_density(positions)
        for s in range(0, n, chunk_size):
            _x = positions[s:s+chunk_size]
            densities.append(
                radiance_field.query_density(_x)
            )
        return torch.cat(densities)

    def rgb_normal_sigma_fn(t_starts, t_ends, ray_indices):
        ray_indices = ray_indices.long()
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        chunk_size = 1024 * 1024
        densities = []
        normals = []
        colors = []
        n = positions.shape[0]
        if n == 0:
            c, d = radiance_field(positions, t_dirs)
            n = torch.zeros_like(c)
            return c, n, d

        for s in range(0, n, chunk_size):
            _x = positions[s:s+chunk_size]
            _t = t_dirs[s:s+chunk_size]
            with torch.enable_grad():
                color, normal, density = radiance_field.forward_with_normal(_x, _t)
            densities.append(density)
            normals.append(normal)
            colors.append(color)

        return torch.cat(colors), torch.cat(normals), torch.cat(densities)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        packed_info, t_starts, t_ends = ray_marching(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            scene_aabb=scene_aabb,
            grid=occupancy_grid,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        rgb, opacity, normal, depth = rendering_with_normal(
            rgb_normal_sigma_fn,
            packed_info,
            t_starts,
            t_ends,
            render_bkgd=render_bkgd,
        )
        chunk_results = [rgb, opacity, normal, depth, len(t_starts)]
        results.append(chunk_results)
    colors, opacities, normals, depths, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        normals.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
    )
