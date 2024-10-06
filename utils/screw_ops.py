import torch

"""
Screw Geometry Utilities

This module provides utility functions for working with screw geometries in PyTorch.
It includes functions to calculate midpoints and angles of screws, as well as cost
functions for comparing predicted and target screw positions.

All functions support batched operations, working on tensors of shape (..., 4) where
the last dimension contains [y_head, x_head, y_tip, x_tip] for each screw.


Example usage:
    import torch
    from screw_utils import screw_head_tip_to_midpoint, screw_head_tip_to_angle

    # Single screw
    screw = torch.tensor([1.0, 0.0, 2.0, 2.0])
    midpoint = screw_head_tip_to_midpoint(screw)
    angle = screw_head_tip_to_angle(screw)

    # Batch of screws
    batch_screws = torch.rand(10, 4)
    batch_midpoints = screw_head_tip_to_midpoint(batch_screws)
    batch_angles = screw_head_tip_to_angle(batch_screws)
"""


def screw_head_tip_to_midpoint(screws: torch.Tensor) -> torch.Tensor:
    """
    Calculates the midpoint of a screw given its head and tip coordinates.

    Args:
        screws (torch.Tensor): Target screw positions of shape (..., 4).

    Returns:
        torch.Tensor: Tensor of shape (..., 2) containing the midpoint coordinates.
    """
    return (screws[..., :2] + screws[..., 2:]) / 2.


def screw_cost_angle(pred_screws, tgt_screws) -> torch.Tensor:
    """
    Calculate the angle difference between predicted and target screw positions.

    Args:
        pred_screws (torch.Tensor): Predicted screw positions of shape (N, 4).
            Each row represents a screw with format [x1, y1, x2, y2],
            where (x1, y1) is the start point and (x2, y2) is the end point.
        tgt_screws (torch.Tensor): Target screw positions of shape (N, 4).
            Each row represents a screw with format [x1, y1, x2, y2],
            where (x1, y1) is the start point and (x2, y2) is the end point.

    Returns:
        torch.Tensor: Angle differences in degrees. Shape is (N).

    """
    # Extract direction vectors
    pred_direction = pred_screws[:, 2:] - pred_screws[:, :2]
    tgt_direction = tgt_screws[:, 2:] - tgt_screws[:, :2]

    # Normalize the direction vectors
    pred_direction_norm = pred_direction / pred_direction.norm(dim=1, keepdim=True)
    tgt_direction_norm = tgt_direction / tgt_direction.norm(dim=1, keepdim=True)

    # Calculate the dot product
    dot_product = (pred_direction_norm * tgt_direction_norm).sum(dim=1)

    # assure that the dot_product is not nan if head and tip predictions are the identical
    dot_product = torch.where(torch.isnan(dot_product), -1.0, dot_product)

    # Clamp dot product to avoid numerical issues with arccos
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Calculate the angle in radians
    angle_diff = torch.acos(dot_product)

    # Convert angle to degrees
    angle_diff_degrees = torch.rad2deg(angle_diff)

    return angle_diff_degrees


def screw_midpoint_to_head_tip(pred_screws: torch.Tensor, tgt_screws: torch.Tensor) -> torch.Tensor:
    """
    Placeholder function to convert screw midpoint representation to head-tip representation.

    Args:
        pred_screws (torch.Tensor): Predicted screw positions of shape (N, 4).
        tgt_screws (torch.Tensor): Target screw positions of shape (M, 4).

    Returns:
        torch.Tensor: Tensor representing screws in head-tip format.
    """
    return screw_cost_head(pred_screws, tgt_screws) + screw_cost_tip(pred_screws, tgt_screws)


def screw_cost_head(pred_screws: torch.Tensor, tgt_screws: torch.Tensor) -> torch.Tensor:
    """
    Compute the L1 cost between predicted and target screw head positions.

    Args:
        pred_screws (torch.Tensor): Predicted screw positions of shape (N, 4).
        tgt_screws (torch.Tensor): Target screw positions of shape (M, 4).

    Returns:
        torch.Tensor: Cost matrix of shape (N, M) containing L1 distances.
    """
    # Compute the L1 cost between screw heads
    return torch.cdist(pred_screws[:, :2], tgt_screws[:, :2], p=1)


def screw_cost_tip(pred_screws: torch.Tensor, tgt_screws: torch.Tensor) -> torch.Tensor:
    """
    Compute the L1 cost between predicted and target screw tip positions.

    Args:
        pred_screws (torch.Tensor): Predicted screw positions of shape (N, 4).
        tgt_screws (torch.Tensor): Target screw positions of shape (M, 4).

    Returns:
        torch.Tensor: Cost matrix of shape (N, M) containing L1 distances.
    """
    # Compute the L1 cost between screw tips
    return torch.cdist(pred_screws[:, 2:], tgt_screws[:, 2:], p=1)


def screw_cost_mid(pred_screws: torch.Tensor, tgt_screws: torch.Tensor) -> torch.Tensor:
    """
    Compute the L1 cost between predicted and target screw midpoints.

    Args:
        pred_screws (torch.Tensor): Predicted screw midpoints of shape (N, 2).
        tgt_screws (torch.Tensor): Target screw midpoints of shape (M, 2).

    Returns:
        torch.Tensor: Cost matrix of shape (N, M) containing L1 distances.
    """
    # Compute the L1 cost between screw midpoints
    return torch.cdist(pred_screws, tgt_screws, p=1)


def screw_cost_iou(pred_screws: torch.Tensor, tgt_screws: torch.Tensor):
    """Compute IoU between predicted and ground truth imaginary screw bounding boxes."""

    # Compute min and max for each screw
    min_pred_screws = torch.min(pred_screws[:, :2], pred_screws[:, 2:])
    max_pred_screws = torch.max(pred_screws[:, :2], pred_screws[:, 2:])
    min_tgt_screws = torch.min(tgt_screws[:, :2], tgt_screws[:, 2:])
    max_tgt_screws = torch.max(tgt_screws[:, :2], tgt_screws[:, 2:])

    ixmin = torch.max(min_pred_screws[:, 0], min_tgt_screws[:, 0])
    iymin = torch.max(min_pred_screws[:, 1], min_pred_screws[:, 1])
    ixmax = torch.min(max_pred_screws[:, 0], max_tgt_screws[:, 0])
    iymax = torch.min(max_pred_screws[:, 1], max_tgt_screws[:, 1])

    iw = torch.clamp(ixmax - ixmin, min=0.)
    ih = torch.clamp(iymax - iymin, min=0.)

    inters = iw * ih

    area_pred = (max_pred_screws[:, 0] - min_pred_screws[:, 0]) * (max_pred_screws[:, 1] - min_pred_screws[:, 1])
    area_gt = (max_tgt_screws[:, 0] - min_tgt_screws[:, 0]) * (max_tgt_screws[:, 1] - min_tgt_screws[:, 1])

    union = area_pred + area_gt - inters

    iou = inters / union
    return iou
