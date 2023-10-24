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
from bake import bake

from examples.utils import (
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid,
    set_random_seed,
)
from nerfacc.estimators.occ_grid import OccGridEstimator
def main(args):
    model_name = f'qff-{args.qff_type}-s{args.min_log2_freq}_{args.max_log2_freq}_{args.num_freqs}-f{args.num_features}-q{args.num_quants}'
    if args.run_name is None:
        args.run_name = f"{args.scene}-{model_name}"
    print("="*40)
    print(args.run_name.center(40))
    print("="*40)

    # training parameters
    max_steps = 50000
    init_batch_size = 1024
    target_sample_batch_size = 1 << 16
    # scene parameters
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    near_plane = 0.0
    far_plane = 1.0e10
    # model parameters
    grid_resolution = 128
    grid_nlvl = 1
    # render parameters
    render_step_size = 5e-3
    weight_decay = (
        1e-5 if args.scene in ["materials", "ficus", "drums"] else 1e-6
    )
    if (args.qff_type == 1) or (args.qff_type == 2):
        weight_decay = weight_decay ** 3
    # setup the dataset
    train_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=args.data_path,
        split=args.train_split,
        num_rays=init_batch_size,
        device=device,
    )
    test_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=args.data_path,
        split="test",
        num_rays=None,
        device=device,
    )

    estimator = OccGridEstimator( roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl).to(device)

    # setup the radiance field we want to train.
    radiance_field = QFFRadianceField(num_quants=args.num_quants,
                                      num_corrs=args.num_corrs,
                                      num_features=args.num_features,
                                      num_freqs=args.num_freqs,
                                      min_log2_freq=args.min_log2_freq,
                                      max_log2_freq=args.max_log2_freq,
                                      qff_type=args.qff_type).to(device)

    num_params = sum([param.numel() for param in radiance_field.parameters()])
    num_enc_params = sum([param.numel() for param in radiance_field.encoder.parameters()])
    print(f'number of parameters: {num_params} / enc params: {num_enc_params}')
    print('-- model --')
    print(radiance_field)

    # optimizer = torch.optim.Adam(radiance_field.parameters(), lr=1e-2, eps=1e-15)
    optimizer = torch.optim.Adam(radiance_field.parameters(), lr=1e-2, eps=1e-15, weight_decay=weight_decay ** 3)
    scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=100
            ),
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    max_steps // 2,
                    max_steps * 3 // 4,
                    max_steps * 9 // 10,
                ],
                gamma=0.33,
            ),
        ]
    )

    lpips_net = LPIPS(net="vgg").to(device)
    lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
    lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

    if args.model_path is not None:
        checkpoint = torch.load(args.model_path)
        radiance_field.load_state_dict(checkpoint["radiance_field_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        estimator.load_state_dict(checkpoint["estimator_state_dict"])
        step = checkpoint["step"]
    else:
        step = 0

    # training
    tic = time.time()
    output_path = Path(args.output_path) / args.run_name
    checkpoint_path = Path(args.checkpoint_path) / args.run_name
    (output_path).mkdir(exist_ok=True, parents=True)
    (checkpoint_path).mkdir(exist_ok=True, parents=True)
    for step in range(max_steps + 1):
        radiance_field.train()
        estimator.train()

        i = torch.randint(0, len(train_dataset), (1,)).item()
        data = train_dataset[i]

        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]

        def occ_eval_fn(x):
            density = radiance_field.query_density(x)
            return density * render_step_size

        # update occupancy grid
        estimator.update_every_n_steps(
            step=step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=1e-2,
        )

        # render
        rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
            radiance_field, estimator, rays,
            # rendering options
            near_plane=near_plane, render_step_size=render_step_size, render_bkgd=render_bkgd,
        )
        if n_rendering_samples == 0:
            continue

        if target_sample_batch_size > 0:
            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = len(pixels)
            num_rays = int( num_rays * (target_sample_batch_size / float(n_rendering_samples)))
            train_dataset.update_num_rays(num_rays)

        # compute loss
        loss = F.smooth_l1_loss(rgb, pixels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 5000 == 0:
            elapsed_time = time.time() - tic
            loss = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(loss) / np.log(10.0)
            print( f"elapsed_time={elapsed_time:.2f}s | step={step} | loss={loss:.5f} | psnr={psnr:.2f} | n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | max_depth={depth.max():.3f} | ")
            Path(args.checkpoint_path).mkdir(exist_ok=True, parents=True)
            bake_path = Path(args.checkpoint_path) / f"{args.run_name}_{step}.qff"
            bake(radiance_field, bake_path)


        # perform testing
        if step > 0 and step % max_steps == 0:
            Path(args.checkpoint_path).mkdir(exist_ok=True, parents=True)
            model_save_path = Path(args.checkpoint_path) / f"{args.run_name}.pth"
            bake_path = Path(args.checkpoint_path) / f"{args.run_name}.qff"
            bake(radiance_field, bake_path)
            torch.save(
                {
                    "step": step,
                    "radiance_field_state_dict": radiance_field.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "estimator_state_dict": estimator.state_dict(),
                },
                model_save_path,
            )

            # evaluation
            radiance_field.eval()
            estimator.eval()

            psnrs = []
            lpips = []
            with torch.no_grad():
                for i in tqdm.tqdm(range(len(test_dataset))):
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
                    psnrs.append(psnr.item())
                    lpips.append(lpips_fn(rgb, pixels).item())

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
    parser.add_argument( "--train_split", type=str, default="train", choices=["train", "trainval"], help="which train split to use",)
    parser.add_argument( "--model_path", type=str, default=None, help="the path of the pretrained model",)
    parser.add_argument( "--scene", type=str, default="chair", choices=NERF_SYNTHETIC_SCENES, help="which scene to use",)
    parser.add_argument( "--test_chunk_size", type=int, default=4096,)
    # checkpoint etc
    parser.add_argument( "--data_path", type=str, default=str(pathlib.Path.cwd() / "data/nerf_synthetic"), help="the root dir of the dataset",)
    parser.add_argument( "--log_path", type=str, help="Path to dataset", default='logs')
    parser.add_argument( "--run_name", type=str)
    parser.add_argument( "--checkpoint_path", type=str, help="Path to dataset", default='checkpoints')
    parser.add_argument( "--output_path", type=str, help="Path to writing outputs", default='outputs')
    # model specific
    parser.add_argument( "--num_quants", type=int, default=80)
    parser.add_argument( "--num_features", type=int, default=4)
    parser.add_argument( "--num_corrs", type=int, default=2)
    parser.add_argument( "--num_freqs", type=int, default=4)
    parser.add_argument( "--min_log2_freq", type=int, default=1)
    parser.add_argument( "--max_log2_freq", type=int, default=6)
    parser.add_argument( "--qff_type", type=int, choices=[0, 1, 2, 3], help="QFF type", default=1)
    args = parser.parse_args()
    main(args)