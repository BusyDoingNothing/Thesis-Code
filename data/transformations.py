import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

class NumpyToPIL:

    def __call__(self, frame):
        return Image.fromarray(frame).convert("RGB")

class TorchToPIL:

    def __call__(self, frame):
        return Image.fromarray(np.rollaxis((frame.numpy() * 255).astype(np.uint8), 0, 3))


class PILToNumpy:

    def __call__(self, frame):
        return np.asarray(frame)


class ToRGB:

    def __call__(self, frame):
        return np.stack([frame] * 3, axis=2)


class Show():
    def __call__(self, frame):
        frame.show()
        return frame

class Affine():

    def __init__(self, sigma):
        self.sigma = sigma

    def _sample_theta(self, batch_size):
        noise = torch.normal(mean=0, std=self.sigma * torch.ones([batch_size, 2, 3]))
        theta = noise + torch.eye(2, 3).view(1, 2, 3)
        return theta

    def __call__(self, imgs):
        if self.sigma > 0:
            imgs = torch.unsqueeze(imgs, dim=0)
            theta = self._sample_theta(imgs.size(0))
            grid = F.affine_grid(theta, imgs.size(), align_corners=False)
            imgs = F.grid_sample(imgs, grid, align_corners=False).squeeze()
        return imgs