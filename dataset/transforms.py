import random
import torch
from torch import tensor
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import RandomCrop
from backbones.BaseBackbone import BaseBackbone


def crop(image, target, region):
    i, j, edge, _ = region
    target = target.copy()
    if "screws" in target:
        target["screws"] = (target["screws"] * image.size(0) - torch.tensor([j, i, j, i])) / edge
    return F.crop(image, *region), target


def flip(image, target, horizontal: bool):
    flip_fn = F.hflip if horizontal else F.vflip
    flip_tensor = torch.tensor([-1, 1, -1, 1] if horizontal else [1, -1, 1, -1])
    add_tensor = torch.tensor([1, 0, 1, 0] if horizontal else [0, 1, 0, 1])

    target = target.copy()
    if "screws" in target:
        target["screws"] = target["screws"] * flip_tensor + add_tensor
    return flip_fn(image), target


def rotate(image, target, angle: int):
    rotated_image = F.rotate(image.unsqueeze(0), angle).squeeze(0)
    target = target.copy()
    if "screws" in target:
        screws = target["screws"]
        if angle == 90:
            screws = screws[:, [1, 0, 3, 2]] * torch.tensor([1, -1, 1, -1]) + torch.tensor([0, 1, 0, 1])
        elif angle == 180:
            screws = screws * -1 + 1
        elif angle == 270:
            screws = screws[:, [1, 0, 3, 2]] * torch.tensor([-1, 1, -1, 1]) + torch.tensor([1, 0, 1, 0])
        target["screws"] = screws
    return rotated_image, target


class V1RandomRotation:
    def __init__(self, multiplier=1):
        self.multiplier = multiplier

    def __call__(self, projection: tensor, target: dict):
        angle = random.choice([0, 90, 180, 270]) * self.multiplier
        target["rotation"] = angle
        return rotate(projection, target, angle)


class V1RandomSizeCrop:
    def __init__(self, min_size: int, max_size: int, backbone: BaseBackbone):
        self.min_size = min_size
        self.max_size = max_size
        self.backbone_image_size = backbone.image_size
        self.resize_transform = transforms.Resize(self.backbone_image_size, antialias=True)

    def __call__(self, projection: tensor, target: dict):
        if "screws" in target:
            screw_min, screw_max = target["screws"].min().item(), target["screws"].max().item()
            screw_min_size = int(min(max(1 - screw_min, screw_max), 1) * projection.size(0))
            edge = random.randint(max(self.min_size, screw_min_size), self.max_size)
        else:
            edge = random.randint(self.min_size, self.max_size)

        region = RandomCrop.get_params(projection, (edge, edge))
        cropped_image, target = crop(projection, target, region)
        target["crop_region"] = list(region)
        return self.resize_transform(cropped_image.unsqueeze(0)).squeeze(0), target


class V1RandomFlip:
    def __init__(self, p=0.5, horizontal=True):
        self.p = p
        self.horizontal = horizontal

    def __call__(self, projection: tensor, target: dict):
        if random.random() < self.p:
            target["h_flip" if self.horizontal else "v_flip"] = True
            return flip(projection, target, self.horizontal)
        target["h_flip" if self.horizontal else "v_flip"] = False
        return projection, target


class V1Resize:
    def __init__(self, backbone: BaseBackbone):
        self.resize_transform = transforms.Resize(backbone.image_size, antialias=True)

    def __call__(self, projection: tensor, target: dict):
        return self.resize_transform(projection.unsqueeze(0)).squeeze(0), target


class V1Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        return f"{self.__class__.__name__}(\n" + "\n".join(f"    {t}" for t in self.transforms) + "\n)"


def get_transformation_matrix(h_flip, v_flip, angle, crop_region, orig_size, device):
    i, j, edge, _ = crop_region

    # Initialize with identity matrix
    matrix = torch.eye(3, device=device)

    # Horizontal flip
    if h_flip:
        matrix = torch.mm(torch.tensor([[-1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=torch.float32, device=device), matrix)

    # Vertical flip
    if v_flip:
        matrix = torch.mm(torch.tensor([[1, 0, 0], [0, -1, 1], [0, 0, 1]], dtype=torch.float32, device=device), matrix)

    # Rotation
    if angle == 90:
        rotation_matrix = torch.tensor([[0, 1, 0], [-1, 0, 1], [0, 0, 1]], dtype=torch.float32, device=device)
    elif angle == 180:
        rotation_matrix = torch.tensor([[-1, 0, 1], [0, -1, 1], [0, 0, 1]], dtype=torch.float32, device=device)
    elif angle == 270:
        rotation_matrix = torch.tensor([[0, -1, 1], [1, 0, 0], [0, 0, 1]], dtype=torch.float32, device=device)
    else:
        rotation_matrix = torch.eye(3, dtype=torch.float32, device=device)

    matrix = torch.mm(rotation_matrix, matrix)

    # Scaling and translation for projection
    scale_translate_matrix = torch.tensor([[orig_size / edge, 0, -j / edge],
                                           [0, orig_size / edge, -i / edge],
                                           [0, 0, 1]], dtype=torch.float32, device=device)
    matrix = torch.mm(scale_translate_matrix, matrix)

    return matrix[None, :, :]


def apply_transformation_matrix_batched(points_batch, transformation_matrix_batch):
    """
    Apply a batch of transformation matrices to a batch of points.

    Args:
        points_batch (torch.Tensor): Batch of points with shape (batch_size, num_points, 2).
        transformation_matrix_batch (torch.Tensor): Batch of transformation matrices with shape (batch_size, 3, 3).

    Returns:
        torch.Tensor: Transformed points with shape (batch_size, num_points, 2).
    """
    batch_size, num_points, num_points_second_view, _ = points_batch.shape
    device = points_batch.device

    # Convert points to homogeneous coordinates
    ones = torch.ones(batch_size, num_points, num_points_second_view, 1, device=device)
    points_homogeneous = torch.cat([points_batch, ones], dim=-1)

    # Apply transformation
    transformed_points_homogeneous = torch.matmul(points_homogeneous, transformation_matrix_batch.transpose(-2, -1))

    return transformed_points_homogeneous[..., :2]


def reverse_transformation_matrix_batched(points_batch, transformation_matrix_batch):
    """
    Apply the inverse of a batch of transformation matrices to a batch of points.

    Args:
        points_batch (torch.Tensor): Batch of points with shape (batch_size, num_points, 2).
        transformation_matrix_batch (torch.Tensor): Batch of transformation matrices with shape (batch_size, 3, 3).

    Returns:
        torch.Tensor: Transformed points with shape (batch_size, num_points, 2).
    """
    batch_size, num_points, lvls, _ = points_batch.shape
    device = points_batch.device
    # Convert points to homogeneous coordinates
    ones = torch.ones(batch_size, num_points, lvls, 1, device=device)
    points_homogeneous = torch.cat([points_batch, ones], dim=-1)
    # Get inverse transformation matrices
    inverse_transformation_matrix_batch = torch.inverse(transformation_matrix_batch)
    # Apply inverse transformation
    transformed_points_homogeneous = torch.matmul(points_homogeneous,
                                                  inverse_transformation_matrix_batch.transpose(-2, -1))
    return transformed_points_homogeneous[..., :2]