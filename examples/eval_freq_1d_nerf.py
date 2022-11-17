"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import math
import os
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from radiance_fields.mlp import *
from utils import render_image, set_random_seed

from tensorboardX import SummaryWriter
from pathlib import Path

from nerfacc import ContractionType, OccupancyGrid

def main(args):
    device = "cuda:0"
    set_random_seed(42)

    render_n_samples = 1024

    # setup the scene bounding box.
    if args.unbounded:
        print("Using unbounded rendering")
        contraction_type = ContractionType.UN_BOUNDED_SPHERE
        # contraction_type = ContractionType.UN_BOUNDED_TANH
        scene_aabb = None
        near_plane = 0.2
        far_plane = 1e4
        render_step_size = 1e-2
    else:
        contraction_type = ContractionType.AABB
        scene_aabb = torch.tensor(args.aabb, dtype=torch.float32, device=device)
        near_plane = None
        far_plane = None
        render_step_size = (
            (scene_aabb[3:] - scene_aabb[:3]).max()
            * math.sqrt(3)
            / render_n_samples
        ).item()

    # setup the radiance field we want to train.
    radiance_field = FreqNeRFRadianceField().to(device)
    radiance_field.load_state_dict(torch.load(args.ckpt_file))

    # setup the dataset
    test_dataset_kwargs = {}
    if args.scene == "garden":
        from datasets.nerf_360_v2 import SubjectLoader

        data_root_fp = args.data_path
        target_sample_batch_size = 1 << 16
        test_dataset_kwargs = {"factor": 4}
        grid_resolution = 128
    else:
        from datasets.nerf_synthetic import SubjectLoader

        data_root_fp = args.data_path
        target_sample_batch_size = 1 << 16
        grid_resolution = 128

    test_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split="test",
        num_rays=None,
        **test_dataset_kwargs,
    )
    test_dataset.images = test_dataset.images.to(device)
    test_dataset.camtoworlds = test_dataset.camtoworlds.to(device)
    test_dataset.K = test_dataset.K.to(device)

    occupancy_grid = OccupancyGrid(
        roi_aabb=args.aabb,
        resolution=grid_resolution,
        contraction_type=contraction_type,
    ).to(device)
    occupancy_grid.load_state_dict(torch.load(args.grid_file))

    # training
    # evaluation
    radiance_field.eval()
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    psnrs = []
    with torch.no_grad():

        for i in tqdm(range(len(test_dataset)), leave=False):
            data = test_dataset[i]
            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]

            # rendering
            rgb, acc, depth, _ = render_image(
                radiance_field,
                occupancy_grid,
                rays,
                scene_aabb,
                # rendering options
                near_plane=None,
                far_plane=None,
                render_step_size=render_step_size,
                render_bkgd=render_bkgd,
                cone_angle=args.cone_angle,
                # test options
                test_chunk_size=args.test_chunk_size,
            )
            mse = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(mse) / np.log(10.0)
            psnrs.append(psnr.item())
            output_image = output_folder / f'{i:06d}.png'
            imageio.imwrite(
                output_image,
                (rgb.cpu().numpy() * 255).astype(np.uint8),
            )
    psnr_avg = sum(psnrs) / len(psnrs)
    print(f"evaluation: psnr_avg={psnr_avg}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_split",
        type=str,
        default="trainval",
        choices=["train", "trainval"],
        help="which train split to use",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="lego",
        choices=[
            # nerf synthetic
            "chair",
            "drums",
            "ficus",
            "hotdog",
            "lego",
            "materials",
            "mic",
            "ship",
            # mipnerf360 unbounded
            "garden",
        ],
        help="which scene to use",
    )
    parser.add_argument(
        "--aabb",
        type=lambda s: [float(item) for item in s.split(",")],
        default="-1.5,-1.5,-1.5,1.5,1.5,1.5",
        help="delimited list input",
    )
    parser.add_argument(
        "--test_chunk_size",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--unbounded",
        action="store_true",
        help="whether to use unbounded rendering",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to dataset",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        help="Path to dataset",
        default='logs'
    )
    parser.add_argument(
        "--run_name",
        type=str,
    )
    parser.add_argument(
        "--output_folder",
        type=str,
    )
    parser.add_argument(
        "--ckpt_file",
        type=str,
    )
    parser.add_argument(
        "--grid_file",
        type=str,
    )
    parser.add_argument("--cone_angle", type=float, default=0.0)
    args = parser.parse_args()
    main(args)