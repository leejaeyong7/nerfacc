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
from radiance_fields.mlp import VanillaNeRFRadianceField, FreqNeRFRadianceField
from utils import render_image, set_random_seed
from pathlib import Path

from tensorboardX import SummaryWriter

from nerfacc import ContractionType, OccupancyGrid

if __name__ == "__main__":

    device = "cuda:0"
    set_random_seed(42)

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
        "--checkpoint_path",
        type=str,
        help="Path to dataset",
        default='checkpoints'
    )
    parser.add_argument(
        "--run_name",
        type=str,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to dataset",
        default='outputs'
    )
    parser.add_argument("--cone_angle", type=float, default=0.0)
    args = parser.parse_args()

    # logger = SummaryWriter(comment=args.log_path+"/" +args.run_name)
    logger = SummaryWriter(logdir=args.log_path + "/" + args.run_name)
    checkpoint_path = Path(args.checkpoint_path)

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
    max_steps = 50000
    val_steps = list(range(0, max_steps, 5000))[1:]

    output_folder = Path(args.output_path)

    grad_scaler = torch.cuda.amp.GradScaler(1)
    radiance_field = VanillaNeRFRadianceField().to(device)
    optimizer = torch.optim.Adam(radiance_field.parameters(), lr=4e-4)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            max_steps // 2,
            max_steps * 3 // 4,
            max_steps * 5 // 6,
            max_steps * 9 // 10,
        ],
        gamma=0.33,
    )

    # setup the dataset
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}
    if args.scene == "garden":
        from datasets.nerf_360_v2 import SubjectLoader

        data_root_fp = args.data_path
        target_sample_batch_size = 1 << 16
        train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
        test_dataset_kwargs = {"factor": 4}
        grid_resolution = 128
    else:
        from datasets.nerf_synthetic import SubjectLoader

        data_root_fp = args.data_path
        target_sample_batch_size = 1 << 16
        grid_resolution = 128

    train_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split=args.train_split,
        num_rays=target_sample_batch_size // render_n_samples,
        **train_dataset_kwargs,
    )

    train_dataset.images = train_dataset.images.to(device)
    train_dataset.camtoworlds = train_dataset.camtoworlds.to(device)
    train_dataset.K = train_dataset.K.to(device)

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

    # training
    step = 0
    tic = time.time()
    global_it = tqdm(range(max_steps), dynamic_ncols=True)
    for epoch in range(1000000):
        num_train_samples = len(train_dataset)
        # train_it = tqdm(range(num_train_samples), total=num_train_samples, dynamic_ncols=True, leave=False)
        for i in range(num_train_samples):
            radiance_field.train()
            data = train_dataset[i]

            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]

            # update occupancy grid
            occupancy_grid.every_n_step(
                step=step,
                occ_eval_fn=lambda x: radiance_field.query_opacity(
                    x, render_step_size
                ),
            )

            # render
            rgb, acc, depth, n_rendering_samples = render_image(
                radiance_field,
                occupancy_grid,
                rays,
                scene_aabb,
                # rendering options
                near_plane=near_plane,
                far_plane=far_plane,
                render_step_size=render_step_size,
                render_bkgd=render_bkgd,
                cone_angle=args.cone_angle,
            )
            if n_rendering_samples == 0:
                continue

            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = len(pixels)
            num_rays = int(
                num_rays
                * (target_sample_batch_size / float(n_rendering_samples))
            )
            train_dataset.update_num_rays(num_rays)
            alive_ray_mask = acc.squeeze(-1) > 0

            # compute loss
            loss = F.smooth_l1_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])

            optimizer.zero_grad()
            # do not unscale it because we are using Adam.
            grad_scaler.scale(loss).backward()
            optimizer.step()
            scheduler.step()

            mse_loss = F.mse_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
            global_it.set_description(
                f"loss={loss:.5f} | " + 
                f"alive_ray_mask={alive_ray_mask.long().sum():05d} | " + 
                f"n_rendering_samples={n_rendering_samples:06d} | num_rays={len(pixels):05d} |"
            )
            logger.add_scalar('train/loss', loss, step)

            if (step in val_steps) or (step == max_steps):
                # evaluation
                radiance_field.eval()

                psnrs = []
                with torch.no_grad():
                    # save to checkpoint
                    torch.save(radiance_field.state_dict(), checkpoint_path / f'{args.scene}_step_{step}_van_model.pth')
                    torch.save(occupancy_grid.state_dict(), checkpoint_path / f'{args.scene}_step_{step}_van_grid.pth')

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

                        (output_folder / args.run_name / f'steps_{step}' / 'images').mkdir(exist_ok=True, parents=True)
                        (output_folder / args.run_name / f'steps_{step}' / 'depths').mkdir(exist_ok=True, parents=True)
                        output_image = output_folder / args.run_name / f'steps_{step}' / 'images' / f'{i:06d}.png'
                        output_depth = output_folder / args.run_name / f'steps_{step}' / 'depths' / f'{i:06d}.png'

                        imageio.imwrite(
                            output_depth,
                            ((depth / depth.max()).cpu().numpy() * 255).astype(np.uint8)
                        )
                        imageio.imwrite(
                            output_image,
                            (rgb.cpu().numpy() * 255).astype(np.uint8),
                        )
                psnr_avg = sum(psnrs) / len(psnrs)
                train_dataset.training = True
                logger.add_scalar('eval/psnr_test', psnr_avg, step)
            global_it.update(1)

            if step == max_steps:
                print("training stops")
                exit()

            step += 1
