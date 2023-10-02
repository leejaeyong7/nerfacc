"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
import sys
sys.path.append('.')
sys.path.append('..')
import argparse
import pathlib
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from datasets.nerf_synthetic import SubjectLoader
from lpips import LPIPS
from radiance_fields.qff import QFFRadianceField
# from radiance_fields.mlp import QFFRadianceField
from pathlib import Path

from examples.utils import (
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid,
    set_random_seed,
)
from nerfacc.estimators.occ_grid import OccGridEstimator
def main(args):
    # scene parameters
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    near_plane = 0.0
    # model parameters
    grid_resolution = 128
    grid_nlvl = 1
    # render parameters
    render_step_size = 5e-3
    test_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=args.data_path,
        split="test",
        num_rays=None,
        device=device,
    )

    ckpt = torch.load(args.model_path)
    model_states = ckpt['radiance_field_state_dict']
    freqs = model_states['encoder.freqs']
    num_freqs = freqs.shape[0]
    min_log2_freq = freqs[0].log2().item()
    max_log2_freq = freqs[-1].log2().item()
    num_out_features = model_states['geom_mlp.0.weight'].shape[1] 
    if 'encoder.qff_plane' in model_states:
        qff_type = 2
        qp = model_states['encoder.qff_plane']
        num_quants = qp.shape[2]
        num_channels = qp.shape[1]
        num_features = num_out_features // (num_freqs * 2)
        num_corrs = num_channels // num_features
    elif 'encoder.qff_volume' in model_states:
        qff_type = 3
        qv = model_states['encoder.qff_volume']
        num_corrs = 1
        num_quants = qv.shape[2]
        num_features = qv.shape[1]
    elif 'encoder.qff_vector' in model_states:
        qff_type = 1
        qv = model_states['encoder.qff_vector']
        num_quants = qv.shape[2]
        num_channels = qv.shape[1]
        num_features = num_out_features // (num_freqs * 2)
        num_corrs = num_channels // num_features
    else:
        raise NotImplementedError



    estimator = OccGridEstimator( roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl).to(device)

    # setup the radiance field we want to train.
    radiance_field = QFFRadianceField(num_quants=num_quants,
                                      num_corrs=num_corrs,
                                      num_features=num_features,
                                      num_freqs=num_freqs,
                                      min_log2_freq=min_log2_freq,
                                      max_log2_freq=max_log2_freq,
                                      qff_type=qff_type).to(device)

    num_params = sum([param.numel() for param in radiance_field.parameters()])
    num_enc_params = sum([param.numel() for param in radiance_field.encoder.parameters()])
    print(f'number of parameters: {num_params} / enc params: {num_enc_params}')
    print('-- model --')
    print(radiance_field)

    lpips_net = LPIPS(net="vgg").to(device)
    lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
    lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

    radiance_field.load_state_dict(ckpt["radiance_field_state_dict"])
    estimator.load_state_dict(ckpt["estimator_state_dict"])
    step = ckpt["step"]

    # evaluation
    radiance_field.eval()
    estimator.eval()
    output_path = Path(args.output_path)

    psnrs = []
    lpips = []
    with torch.no_grad():
        iterator =tqdm.tqdm(range(len(test_dataset))) 
        for i in iterator:
            data = test_dataset[i]
            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]

            # rendering
            rgb, acc, depth, _ = render_image_with_occgrid(
                radiance_field, estimator, rays,
                # rendering options
                near_plane=near_plane, render_step_size=render_step_size, render_bkgd=render_bkgd,
                # test options
                test_chunk_size=args.test_chunk_size,
            )
            mse = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(mse) / np.log(10.0)
            psnr_val = psnr.item()
            psnrs.append(psnr_val)
            lpips_val = lpips_fn(rgb, pixels).item()
            lpips.append(lpips_val)
            iterator.set_description(f"PSNR: {psnr_val} | LPIPS: {lpips_val}")

            (output_path / args.run_name / f'steps_{step}' / 'images').mkdir(exist_ok=True, parents=True)
            (output_path / args.run_name / f'steps_{step}' / 'depths').mkdir(exist_ok=True, parents=True)
            output_image = output_path / args.run_name / f'steps_{step}' / 'images' / f'{i:06d}.png'
            output_depth = output_path / args.run_name / f'steps_{step}' / 'depths' / f'{i:06d}.png'

            imageio.imwrite( output_depth, ((depth / depth.max()).cpu().numpy() * 255).astype(np.uint8))
            imageio.imwrite( output_image, (rgb.cpu().numpy() * 255).astype(np.uint8),)

    psnr_avg = sum(psnrs) / len(psnrs)
    lpips_avg = sum(lpips) / len(lpips)
    with open(output_path / args.run_name / f'results.txt', 'w') as f:
        f.write(f"psnr:{psnr_avg}\n")
        f.write(f"lpips_avg={lpips_avg}")
    print(f"psnr: {psnr_avg}, lpips: {lpips_avg}")

if __name__ == "__main__":
    device = "cuda:0"
    set_random_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument( "--model_path", type=str, default=None, help="the path of the pretrained model",)
    parser.add_argument( "--scene", type=str, default="chair", choices=NERF_SYNTHETIC_SCENES, help="which scene to use",)
    parser.add_argument( "--test_chunk_size", type=int, default=4096,)
    parser.add_argument( "--run_name", type=str, default=None)

    # checkpoint etc
    parser.add_argument( "--data_path", type=str, default=str(pathlib.Path.cwd() / "data/nerf_synthetic"), help="the root dir of the dataset",)
    parser.add_argument( "--output_path", type=str, help="Path to writing outputs", default='outputs')
    args = parser.parse_args()
    main(args)