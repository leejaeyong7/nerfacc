"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import collections
import json
import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import re

from .utils import Rays

def read_pfm(pfm_file_path: str)-> torch.Tensor:
    """parses PFM file into torch float tensor

    :param pfm_file_path: path like object that contains full path to the PFM file

    :returns: parsed PFM file of shape CxHxW
    """
    color = None
    width = None
    height = None
    scale = None
    data_type = None
    with open(pfm_file_path, 'rb') as file:
        header = file.readline().decode('UTF-8').rstrip()

        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))

        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        # scale = float(file.readline().rstrip())
        scale = float((file.readline()).decode('UTF-8').rstrip())
        if scale < 0: # little-endian
            data_type = '<f'
        else:
            data_type = '>f' # big-endian
        data_string = file.read()
        data = np.fromstring(data_string, data_type)
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.ascontiguousarray(np.flip(data, 0))
    return data.reshape(height, width, -1)

def _load_renderings(root_fp: str, subject_id: str):
    """Load images from disk."""
    if not root_fp.startswith("/"):
        # allow relative path. e.g., "./data/nerf_synthetic/"
        root_fp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            root_fp,
        )

    data_dir = os.path.join(root_fp, subject_id)
    with open(
        os.path.join(data_dir, "transforms.json"), "r"
    ) as fp:
        meta = json.load(fp)
    images = []
    normals = []
    depths = []
    camtoworlds = []

    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        fname = os.path.join(data_dir, frame["file_path"])
        nname = os.path.join(data_dir, frame["normal_file_path"])
        dname = os.path.join(data_dir, frame["depth_file_path"])

        rgb = imageio.imread(fname)
        a = np.ones_like(rgb[..., :1]) * 255
        rgba = np.concatenate((rgb, a), -1)
        normal = (imageio.imread(nname) / 255.0) * 2 - 1
        depth = read_pfm(dname)

        camtoworlds.append(frame["transform_matrix"])
        images.append(rgba)
        normals.append(normal)
        depths.append(depth)

    images = np.stack(images, axis=0)
    normals = np.stack(normals, axis=0)
    depths = np.stack(depths, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)

    h, w = images.shape[1:3]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

    return images, normals, depths, camtoworlds, focal


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SUBJECT_IDS = [
        "courtyard",
        "delivery_area",
        "electro",
        "facade",
        "kicker",
        "meadow",
        "office",
        "pipes",
        "playground",
        "relief",
        "relief_2",
        "terrace",
        "terrains"
    ]

    WIDTH, HEIGHT = 960,640
    NEAR, FAR = 0.001, 6.0
    OPENGL_CAMERA = True

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        num_rays: int = None,
        near: float = None,
        far: float = None,
        training: bool = True,
        batch_over_images: bool = True,
    ):
        super().__init__()
        assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        self.num_rays = num_rays
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        self.training = training
        self.batch_over_images = batch_over_images
        self.images, self.normals, self.depths, self.camtoworlds, self.focal = _load_renderings( root_fp, subject_id)
        self.images = torch.from_numpy(self.images).to(torch.uint8)
        self.normals = torch.from_numpy(self.normals).to(torch.float32)
        self.depths = torch.from_numpy(self.depths).to(torch.float32)
        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
        self.K = torch.tensor(
            [
                [self.focal, 0, self.WIDTH / 2.0],
                [0, self.focal, self.HEIGHT / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )  # (3, 3)
        assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)

    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]
        pixels, alpha = torch.split(rgba, [3, 1], dim=-1)

        # just use white during inference
        color_bkgd = torch.ones(3, device=self.images.device)

        pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays

        if self.training:
            if self.batch_over_images:
                image_id = torch.randint(
                    0,
                    len(self.images),
                    size=(num_rays,),
                    device=self.images.device,
                )
            else:
                image_id = [index]
            x = torch.randint(
                0, self.WIDTH, size=(num_rays,), device=self.images.device
            )
            y = torch.randint(
                0, self.HEIGHT, size=(num_rays,), device=self.images.device
            )
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH, device=self.images.device),
                torch.arange(self.HEIGHT, device=self.images.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        rgba = self.images[image_id, y, x] / 255.0  # (num_rays, 4)
        normal = self.normals[image_id, y, x] # (num_rays, 3)
        depth = self.depths[image_id, y, x] # (num_rays, 1)

        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5)
                    / self.K[1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgba = torch.reshape(rgba, (num_rays, 4))
            normal = torch.reshape(normal, (num_rays, 3))
            depth = torch.reshape(depth, (num_rays, 1))
        else:
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))
            rgba = torch.reshape(rgba, (self.HEIGHT, self.WIDTH, 4))
            normal = torch.reshape(rgba, (self.HEIGHT, self.WIDTH, 3))
            depth = torch.reshape(rgba, (self.HEIGHT, self.WIDTH, 1))

        rays = Rays(origins=origins, viewdirs=viewdirs)
        indices = torch.ones_like(depth) * index
        return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "normal": normal,  # [h, w, 3] or [num_rays, 3]
            "depth": depth,  # [h, w, 3] or [num_rays, 1]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "indices": indices
        }
