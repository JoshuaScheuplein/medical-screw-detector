from dataclasses import dataclass
import torch
from typing import Tuple


@dataclass
class PointTransformation:
    h_flip: bool = False
    v_flip: bool = False
    angle: int = 0
    crop_region: Tuple[int, int, int, int] = (0, 0, 976, 976)
    orig_size: int = 976

    def __post_init__(self):
        if self.angle not in [0, 90, 180, 270]:
            raise ValueError("Angle must be 0, 90, 180, or 270 degrees")

        if len(self.crop_region) != 4:
            raise ValueError("Crop region must be a tuple of 4 integers")

        if any(not isinstance(x, int) for x in self.crop_region):
            raise ValueError("Crop region values must be integers")

        if self.orig_size <= 0:
            raise ValueError("Original size must be a positive integer")

    def project(self, points):
        return project_points(points, self.h_flip, self.v_flip,
                              self.angle, self.crop_region, self.orig_size)

    def reproject(self, points):
        return reproject_points(points, self.h_flip, self.v_flip,
                                self.angle, self.crop_region, self.orig_size)


def project_points(points, h_flip, v_flip, angle, crop_region, orig_size=976):
    i, j, edge, _ = crop_region

    if h_flip:
        points = points * torch.as_tensor([-1, 1], device=points.device) + torch.as_tensor([1, 0], device=points.device)

    if v_flip:
        points = points * torch.as_tensor([1, -1], device=points.device) + torch.as_tensor([0, 1], device=points.device)

    if angle == 90:
        points = points[..., (1, 0)] * torch.as_tensor([1, -1], device=points.device) + torch.as_tensor([0, 1], device=points.device)
    elif angle == 180:
        points = points * torch.as_tensor([-1, -1], device=points.device) + torch.as_tensor([1, 1], device=points.device)
    elif angle == 270:
        points = points[..., (1, 0)] * torch.as_tensor([-1, 1], device=points.device) + torch.as_tensor([1, 0], device=points.device)

    points = (points * orig_size - torch.as_tensor([j, i], device=points.device)) / edge

    return points


def reproject_points(points, h_flip, v_flip, angle, crop_region, orig_size=976):
    i, j, edge, _ = crop_region

    points = (points * edge + torch.as_tensor([j, i], device=points.device)) / orig_size

    if angle == 90:
        points = points[..., (1, 0)] * torch.as_tensor([-1, 1], device=points.device) + torch.as_tensor([1, 0], device=points.device)
    elif angle == 180:
        points = points * torch.as_tensor([-1, -1], device=points.device) + torch.as_tensor([1, 1], device=points.device)
    elif angle == 270:
        points = points[..., (1, 0)] * torch.as_tensor([1, -1], device=points.device) + torch.as_tensor([0, 1], device=points.device)

    if v_flip:
        points = points * torch.as_tensor([1, -1], device=points.device) + torch.as_tensor([0, 1], device=points.device)

    if h_flip:
        points = points * torch.as_tensor([-1, 1], device=points.device) + torch.as_tensor([1, 0], device=points.device)

    return points