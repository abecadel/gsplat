import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
from PIL import Image
from torch import Tensor, optim
from tqdm import tqdm

from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians


class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
            self,
            gt_image: Tensor,
            mask_image: Tensor,
            num_points: int = 2000,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)
        self.mask_image = mask_image.to(device=self.device)
        self.mask = mask_image[:, :, 1] == True

        self.background_image = torch.zeros_like(mask_image).to(device=self.device)
        self.num_points = num_points

        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)

        self._init_gaussians()

    def _init_gaussians(self):
        """Random gaussians"""
        bd = 2

        self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        # self.means[:, 0] = torch.FloatTensor(self.num_points).uniform_(-1.0, -0.5) - 2.0
        # self.means[:, 0] = torch.ones(self.num_points) * -10.0
        # self.means[:, 1] = torch.FloatTensor(self.num_points).uniform_(0.0, 0.1)
        # self.means[:, 2] = torch.ones(self.num_points) + 0.5
        # self.means[:, 2] = torch.FloatTensor(self.num_points).uniform_(0.9, 1.0)

        self.scales = torch.rand(self.num_points, 3, device=self.device)
        d = 3
        self.rgbs = torch.rand(self.num_points, d, device=self.device)

        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)
        # u = torch.ones(self.num_points, 1, device=self.device) - 1.0
        # v = torch.ones(self.num_points, 1, device=self.device) - 1.0
        # w = torch.ones(self.num_points, 1, device=self.device) - 1.0

        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        self.opacities = torch.ones((self.num_points, 1), device=self.device)

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.background = torch.zeros(d, device=self.device)

        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False

    def train(
            self,
            iterations: int = 1000,
            lr: float = 0.01,
            save_imgs: bool = False,
            B_SIZE: int = 14,
            debug: bool = False,
            mask_penalty: float = 1.0
    ):
        optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats], lr
        )
        mse_loss = torch.nn.MSELoss()
        frames = []
        frames_points_frames = []
        mask_intersection_frames = []
        times = [0] * 3  # project, rasterize, backward
        B_SIZE = 16

        pbar = tqdm(range(iterations))
        for iter in pbar:
            start = time.time()
            (
                xys,
                depths,
                radii,
                conics,
                compensation,
                num_tiles_hit,
                cov3d,
            ) = project_gaussians(
                self.means,
                self.scales,
                1,
                self.quats,
                self.viewmat,
                self.focal,
                self.focal,
                self.W / 2,
                self.H / 2,
                self.H,
                self.W,
                B_SIZE,
            )
            torch.cuda.synchronize()
            times[0] += time.time() - start
            start = time.time()
            out_img = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                torch.sigmoid(self.rgbs),
                torch.sigmoid(self.opacities),
                self.H,
                self.W,
                B_SIZE,
                self.background,
            )[..., :3]
            torch.cuda.synchronize()
            times[1] += time.time() - start
            image_loss = mse_loss(out_img, self.gt_image)

            mask_intersection = out_img.clone()
            mask_intersection[self.mask] = 0  # hardcoded background
            mask_loss = mse_loss(mask_intersection, self.background_image) * mask_penalty

            optimizer.zero_grad()
            start = time.time()
            loss = image_loss + mask_loss
            loss.backward()
            torch.cuda.synchronize()
            times[2] += time.time() - start
            optimizer.step()
            pbar.set_description(
                f"Image loss: {image_loss.item():.8f} Mask loss: {mask_loss.item():.8f} Total loss: {loss.item():.8f}")

            if debug or (save_imgs and iter % 100 == 0):
                gaussian_points = xys.detach().cpu().numpy()
                gaussian_points[:, 0] = gaussian_points[:, 0] * (255 / self.W)
                gaussian_points[:, 1] = gaussian_points[:, 1] * (255 / self.H)
                gaussian_points = gaussian_points.astype(np.uint8)

                frame = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
                frame[gaussian_points[:, 0], gaussian_points[:, 1]] = [255, 0, 0]
                frames.append(frame)
                out_dir = os.path.join(os.getcwd(), "renders")
                Image.fromarray(frame).save(f"{out_dir}/{iter}.jpeg")

                # save gaussian positions
                gaussian_points_arr = np.ones_like(frame) * 255
                gaussian_points_arr[gaussian_points[:, 0], gaussian_points[:, 1]] = [
                    255,
                    0,
                    0,
                ]
                frames_points_frames.append(gaussian_points_arr)
                Image.fromarray(gaussian_points_arr).save(
                    f"{out_dir}/{iter}-gaussian-points.jpeg"
                )

                # mask intersection
                mask_intersection_frame = (mask_intersection.detach().cpu().numpy() * 255).astype(np.uint8)
                mask_intersection_frame[gaussian_points[:, 0], gaussian_points[:, 1]] = [255, 0, 0]
                mask_intersection_frames.append(mask_intersection_frame)
                out_dir = os.path.join(os.getcwd(), "renders")
                Image.fromarray(mask_intersection_frame).save(f"{out_dir}/{iter}-mask-intersection.jpeg")

                # background
                background = (self.background_image.detach().cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(background).save(f"{out_dir}/{iter}-mask-background.jpeg")

        if save_imgs:
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "renders")
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )

            # save points them as a gif with PIL
            frames_points_frames = [Image.fromarray(frame) for frame in frames_points_frames]
            out_dir = os.path.join(os.getcwd(), "renders")
            os.makedirs(out_dir, exist_ok=True)
            frames_points_frames[0].save(
                f"{out_dir}/training-points.gif",
                save_all=True,
                append_images=frames_points_frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )

            # save mask intersections as a gif with PIL
            mask_intersection_frames = [Image.fromarray(frame) for frame in mask_intersection_frames]
            out_dir = os.path.join(os.getcwd(), "renders")
            os.makedirs(out_dir, exist_ok=True)
            mask_intersection_frames[0].save(
                f"{out_dir}/training-mask-intersection.gif",
                save_all=True,
                append_images=mask_intersection_frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )
        print(
            f"Total(s):\nProject: {times[0]:.3f}, Rasterize: {times[1]:.3f}, Backward: {times[2]:.3f}"
        )
        print(
            f"Per step(s):\nProject: {times[0] / iterations:.5f}, Rasterize: {times[1] / iterations:.5f}, Backward: {times[2] / iterations:.5f}"
        )


def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor


def main(
        height: int = 256,
        width: int = 256,
        num_points: int = 100000,
        save_imgs: bool = True,
        img_path: Optional[Path] = None,
        mask_path: Optional[Path] = None,
        iterations: int = 1000,
        lr: float = 0.01,
        debug: bool = False,
        mask_penalty: float = 1.0
) -> None:
    if img_path:
        gt_image = image_path_to_tensor(img_path)
    else:
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2:, width // 2:, :] = torch.tensor([0.0, 0.0, 1.0])

    if mask_path:
        mask_image = image_path_to_tensor(mask_path)
    else:
        mask_image = torch.ones((height, width, 3)) * 1.0

    trainer = SimpleTrainer(
        gt_image=gt_image, mask_image=mask_image, num_points=num_points
    )
    trainer.train(iterations=iterations, lr=lr, save_imgs=save_imgs, debug=debug, mask_penalty=mask_penalty)


if __name__ == "__main__":
    tyro.cli(main)
