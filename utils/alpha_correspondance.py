import json

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.patches import Polygon
from tifffile import tifffile

from utils.data_normalization import neglog_normalize

def project_to_second_view_orig(P0, P1, reference_points):

    # Construct 0-degree epipolar plane from camera positions and isocenter
    C0 = -torch.linalg.inv(P0[:3, :3]) @ P0[:3, 3]
    C1 = -torch.linalg.inv(P1[:3, :3]) @ P1[:3, 3]

    reference_points_flatten = torch.flatten(reference_points, start_dim=0, end_dim=-2)
    reference_points_x = reference_points_flatten[:, 0]
    reference_points_y = reference_points_flatten[:, 1]

    # Calculate e1
    e1 = P0 @ torch.cat([C1, torch.ones(1, device=P0.device)])
    e1 = e1[:2] / e1[2]
    e1 = e1[[1, 0]]

    # Calculate i1
    i1 = P0 @ torch.tensor([0, 0, 0, 1.0], device=P0.device)
    i1 = i1[:2] / i1[2]
    i1 = i1[[1, 0]]

    # Calculate b1
    b1 = i1 - e1
    b1 /= torch.norm(b1)

    # Calculate d
    d = torch.stack((reference_points_x - e1[0], reference_points_y - e1[1]), dim=-1)
    d /= torch.norm(d, dim=-1, keepdim=True)

    # Calculate e0
    e0 = P1 @ torch.cat([C0, torch.ones(1, device=P1.device)])
    e0 = e0[:2] / e0[2]
    e0 = e0[[1, 0]]

    # Calculate i0
    i0 = P1 @ torch.tensor([0, 0, 0, 1.0], device=P1.device)
    i0 = i0[:2] / i0[2]
    i0 = i0[[1, 0]]

    # Calculate b0
    b0 = i0 - e0
    b0 /= torch.norm(b0)

    # Normalize v1 and v2
    b1 = b1 / torch.norm(b1)

    rot_vecs = []
    for d_ind in d:
        # Calculate the dot product
        d_ind = d_ind / torch.norm(d_ind)
        dot_product = torch.dot(b1, d_ind)
        cos_theta = torch.clamp(dot_product, -1.0, 1.0)
        # Calculate the angle in radians
        angle_radians_old = torch.acos(cos_theta)

        # Ensure the value is within the valid range for arccos
        angle_radians = torch.atan2(d_ind[0] * b1[1] - d_ind[1] * b1[0], torch.dot(d_ind, b1))

        # Calculate the rotation matrix
        rotation_matrix = torch.tensor([
            [torch.cos(angle_radians), -torch.sin(angle_radians)],
            [torch.sin(angle_radians), torch.cos(angle_radians)]
        ])

        # Rotate the original vector
        rotated_vector = torch.matmul(rotation_matrix, b0)
        rot_vecs.append(rotated_vector)

    rotated = torch.stack(rot_vecs)
    rotated[:] = rotated[:] / rotated[:, :1]

    x0_pts = e0 - (e0[:1] / rotated[:, :1]) * rotated

    return x0_pts.unsqueeze(dim=0).unsqueeze(dim=0), rotated.unsqueeze(dim=0).unsqueeze(dim=0)


def project_to_second_view_fixed_trigonometric(P0, P1, reference_points):
    # Assuming P0 and P1 are of shape (batch_size, 3, 4)
    # and reference_points is of shape (batch_size, query_len, num_points, 2)
    batch_size, Len_q, lvl, _ = reference_points.shape

    # Construct 0-degree epipolar plane from camera positions and isocenter
    C0 = -torch.linalg.inv(P0[:, :3, :3]) @ P0[:, :3, 3].unsqueeze(-1)
    C1 = -torch.linalg.inv(P1[:, :3, :3]) @ P1[:, :3, 3].unsqueeze(-1)

    reference_points_flatten = torch.flatten(reference_points, start_dim=1, end_dim=-2)

    # Calculate e0 and e1
    e0 = P1 @ torch.cat([C0, torch.ones(batch_size, 1, 1, device=P1.device)], dim=1)
    e0 = e0.squeeze(-1)
    e0 = e0[:, :2] / e0[:, 2:3]
    # swap x and y
    e0 = e0[:, [1, 0]].unsqueeze(dim=1)

    # Calculate e1
    e1 = P0 @ torch.cat([C1, torch.ones(batch_size, 1, 1, device=P0.device)], dim=1)
    e1 = e1.squeeze(-1)
    e1 = e1[:, :2] / e1[:, 2:3]
    # swap x and y
    e1 = e1[:, [1, 0]].unsqueeze(dim=1)

    # Calculate i0 and i1
    i0 = P1 @ torch.tensor([0, 0, 0, 1.0], device=P1.device)
    i0 = i0[:, :2] / i0[:, 2]
    i0 = i0[:, [1, 0]].unsqueeze(dim=1)

    i1 = P0 @ torch.tensor([0, 0, 0, 1.0], device=P0.device)
    i1 = i1[:, :2] / i1[:, 2]
    i1 = i1[:, [1, 0]].unsqueeze(dim=1)

    # Calculate b0 and b1
    b0 = i0 - e0
    b0 /= torch.norm(b0, dim=-1, keepdim=True)

    b1 = i1 - e1
    b1 /= torch.norm(b1, dim=-1, keepdim=True)

    # Calculate d
    d = reference_points_flatten - e1
    d /= torch.norm(d, dim=-1, keepdim=True)

    # Calculate the angle in radians (vectorized)
    cross_product = d[:, :, 0] * b1[:, :, 1] - d[:, :, 1] * b1[:, :, 0]
    dot_product = torch.sum(d * b1, dim=-1)
    angle_radians = torch.atan2(cross_product, dot_product)

    # Calculate the rotation matrices (vectorized)
    cos_angle = torch.cos(angle_radians)
    sin_angle = torch.sin(angle_radians)
    rotation_matrices = torch.stack([
        torch.stack([cos_angle, -sin_angle], dim=-1),
        torch.stack([sin_angle, cos_angle], dim=-1)
    ], dim=-2)

    # Rotate the original vector (vectorized)
    rotated = torch.einsum('bnij,bkj->bnik', rotation_matrices, b0)
    rotated = rotated.squeeze(3)
    rotated[:, :] = rotated[:, :] / rotated[:, :, :1]

    x0_pts = e0 - e0[:, :1, :1] * rotated

    x0_pts = x0_pts.view(batch_size, Len_q, lvl, 2)
    rotated = rotated.view(batch_size, Len_q, lvl, 2)

    return x0_pts, rotated


def project_to_second_view_fixed(P0, P1, reference_points):
    # Assuming P0 and P1 are of shape (batch_size, 3, 4)
    # and reference_points is of shape (batch_size, query_len, num_points, 2)
    batch_size, Len_q, lvl, _ = reference_points.shape

    # Construct 0-degree epipolar plane from camera positions and isocenter
    C0 = -torch.linalg.inv(P0[:, :3, :3]) @ P0[:, :3, 3].unsqueeze(-1)
    C1 = -torch.linalg.inv(P1[:, :3, :3]) @ P1[:, :3, 3].unsqueeze(-1)

    reference_points_flatten = torch.flatten(reference_points, start_dim=1, end_dim=-2)

    # Calculate e0 and e1
    e0 = P1 @ torch.cat([C0, torch.ones(batch_size, 1, 1, device=P1.device)], dim=1)
    e0 = e0.squeeze(-1)
    e0 = e0[:, :2] / e0[:, 2:3]
    # swap x and y
    e0 = e0[:, [1, 0]].unsqueeze(dim=1)

    # Calculate e1
    e1 = P0 @ torch.cat([C1, torch.ones(batch_size, 1, 1, device=P0.device)], dim=1)
    e1 = e1.squeeze(-1)
    e1 = e1[:, :2] / e1[:, 2:3]
    # swap x and y
    e1 = e1[:, [1, 0]].unsqueeze(dim=1)

    # Calculate i0 and i1
    i0 = P1 @ torch.tensor([0, 0, 0, 1.0], device=P1.device)
    i0 = i0[:, :2] / i0[:, 2:3]
    i0 = i0[:, [1, 0]].unsqueeze(dim=1)

    i1 = P0 @ torch.tensor([0, 0, 0, 1.0], device=P0.device)
    i1 = i1[:, :2] / i1[:, 2:3]
    i1 = i1[:, [1, 0]].unsqueeze(dim=1)

    # Calculate b0 and b1
    b0 = i0 - e0
    b0 /= torch.norm(b0, dim=-1, keepdim=True)

    b1 = i1 - e1
    b1 /= torch.norm(b1, dim=-1, keepdim=True)

    # Calculate d
    d = reference_points_flatten - e1
    d /= torch.norm(d, dim=-1, keepdim=True)

    cross_product = d[:, :, 0] * b1[:, :, 1] - d[:, :, 1] * b1[:, :, 0]
    dot_product = torch.sum(d * b1, dim=-1)

    # Calculate cos and sin of the angle using dot and cross products
    cos_angle = dot_product
    sin_angle = cross_product

    # Construct the rotation matrices
    rotation_matrices = torch.zeros(batch_size, Len_q * lvl, 2, 2, device=b0.device)
    rotation_matrices[:, :, 0, 0] = cos_angle
    rotation_matrices[:, :, 0, 1] = -sin_angle
    rotation_matrices[:, :, 1, 0] = sin_angle
    rotation_matrices[:, :, 1, 1] = cos_angle

    # Rotate the original vector (vectorized)
    rotated = torch.einsum('bnij,bkj->bnik', rotation_matrices, b0)
    rotated = rotated.squeeze(dim=3)
    # rotated = torch.einsum('bij,bjk->bik', rotation_matrices, b0)
    rotated[:, :] = rotated[:, :] / rotated[:, :, :1]

    x0_pts = e0 - e0[:, :1, :1] * rotated

    x0_pts = x0_pts.view(batch_size, Len_q, lvl, 2)
    rotated = rotated.view(batch_size, Len_q, lvl, 2)

    return x0_pts, rotated


def project_to_second_view(P0, P1, reference_points):
    # Assuming P0 and P1 are of shape (batch_size, 3, 4)
    # and reference_points is of shape (batch_size, query_len, num_points, 2)
    batch_size, Len_q, lvl, _ = reference_points.shape

    # Construct 0-degree epipolar plane from camera positions and isocenter
    C0 = -torch.linalg.inv(P0[:, :3, :3]) @ P0[:, :3, 3].unsqueeze(-1)
    C1 = -torch.linalg.inv(P1[:, :3, :3]) @ P1[:, :3, 3].unsqueeze(-1)

    reference_points_flatten = torch.flatten(reference_points, start_dim=1, end_dim=-2)

    # Calculate e0
    e0 = P1 @ torch.cat([C0, torch.ones(batch_size, 1, 1, device=P1.device)], dim=1)
    e0 = e0.squeeze(-1)
    e0 = e0[:, :2] / e0[:, 2:3]
    # swap x and y
    e0 = e0[:, [1, 0]]
    # unsqueeze to match the shape of reference_points_flatten
    e0 = e0.unsqueeze(1)

    # Calculate e1
    e1 = P0 @ torch.cat([C1, torch.ones(batch_size, 1, 1, device=P0.device)], dim=1)
    e1 = e1.squeeze(-1)
    e1 = e1[:, :2] / e1[:, 2:3]
    # swap x and y
    e1 = e1[:, [1, 0]]
    # unsqueeze to match the shape of reference_points_flatten
    e1 = e1.unsqueeze(1)

    # Calculate d
    d = reference_points_flatten - e1
    d /= d[:, :, :1]
    d[..., 1] *= -1.

    x0_pts = e0 - (e0[:, :, :1] / d[:, :, :1]) * d

    x0_pts = x0_pts.view(batch_size, Len_q, lvl, 2)
    d = d.view(batch_size, Len_q, lvl, 2)

    return x0_pts, d

def verify_projection():
    # base = "E:\MA_Data\V1-1to3objects-400projections-circular"
    dir = "Elbow03_le_512x512x512_2"

    predictions = json.load(open(f"{base}/{dir}/labels.json"))

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"]
    })

    for i in range(0, 220, 10):

        view_id = i
        orthogonal_view_id = view_id + 180

        x_max = 976
        colors = list(TABLEAU_COLORS.values())

        with tifffile.TiffFile(f"{base}/{dir}/projections.tiff") as projection_file:
            view = projection_file.asarray(key=slice(view_id, (view_id + 1)))
            view = neglog_normalize(view)

            orthogonal_view = projection_file.asarray(key=slice(orthogonal_view_id, (orthogonal_view_id + 1)))
            orthogonal_view = neglog_normalize(orthogonal_view)

        view_prediction = predictions["landmarks2d"][f"view_{view_id}"]
        view_P = torch.tensor(view_prediction["P_pfw"]).reshape(3, 4)
        view_points = []
        for screw in view_prediction['objects'].values():
            view_points.append(screw[0][::-1])
            view_points.append(screw[1][::-1])

        view_points = torch.tensor(view_points)

        orthogonal_view_prediction = predictions["landmarks2d"][f"view_{orthogonal_view_id}"]
        orthogonal_view_P = torch.tensor(orthogonal_view_prediction["P_pfw"]).reshape(3, 4)

        view_points = view_points.unsqueeze(dim=0).unsqueeze(dim=0)
        view_P = view_P.unsqueeze(dim=0)
        orthogonal_view_P = orthogonal_view_P.unsqueeze(dim=0)

        # ts, ms = project_to_second_view_(view_P, orthogonal_view_P, view_points)

        ts, ms = project_to_second_view_fixed(view_P, orthogonal_view_P, view_points)

        # Create the first image (overlayed arrays)
        fig = plt.figure(figsize=(28, 13))

        ax1 = fig.add_subplot(1, 2, 1)

        # Plot array1 in red and array2 in blue
        ax1.imshow(view, cmap='gray')
        for screw_i, (x, y) in enumerate(view_points[0, 0]):
            screw_i = screw_i // 2
            ax1.plot(x, y, 'x', color=colors[screw_i], markersize=25, markeredgewidth=4, label='Points')

        ax1.set_title(rf"original view ($\theta = {view_id}$)")
        plt.tight_layout()

        ax2 = fig.add_subplot(1, 2, 2)

        # Plot array1 in red and array2 in blue
        ax2.imshow(orthogonal_view, cmap='gray')

        ts = ts[0, 0].cpu().numpy()
        ms = ms[0, 0].cpu().numpy()

        for screw_i, (t, m) in enumerate(zip(ts, ms)):
            screw_i = screw_i // 2
            ax2.plot([t[0], t[0] + m[0] * x_max], [t[1], t[1] + m[1] * x_max],
                     color=colors[screw_i], linestyle='--', lw=3)

            y_offset = 15

            vertices = [
                (0, t[1] + y_offset),
                (976, t[1] + m[1] * x_max + y_offset),
                (976, t[1] + m[1] * x_max - y_offset),
                (0, t[1] - y_offset)
            ]

            # Create a Polygon patch
            quadrilateral = Polygon(vertices, closed=True, linewidth=1, edgecolor=colors[screw_i], facecolor=colors[screw_i], alpha=0.2)
            ax2.add_patch(quadrilateral)

        plt.xlim(0, 976)
        plt.ylim(976, 0)

        ax2.set_title(rf"auxiliary view ($\theta = {orthogonal_view_id // 2}$)")

        plt.show()


def projection_in_image(view_id, orthogonal_view_id):
    # base = "E:\MA_Data\V1-1to3objects-400projections-circular"
    dir = "Elbow03_le_512x512x512_2"

    predictions = json.load(open(f"{base}/{dir}/labels.json"))

    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "serif",
    #     "font.serif": ["Computer Modern"]
    # })

    x_max = 976
    colors = list(TABLEAU_COLORS.values())

    with tifffile.TiffFile(f"{base}/{dir}/projections.tiff") as projection_file:
        view = projection_file.asarray(key=slice(view_id, (view_id + 1)))
        # view = neglog_normalize(view)

    orthogonal_view_prediction = predictions["landmarks2d"][f"view_{orthogonal_view_id}"]
    orthogonal_view_P = torch.tensor(orthogonal_view_prediction["P_pfw"]).reshape(3, 4).unsqueeze(dim=0)

    orthogonal_view_points = []
    for screw in orthogonal_view_prediction['objects'].values():
        orthogonal_view_points.append(screw[0][::-1])
        orthogonal_view_points.append(screw[1][::-1])

    orthogonal_view_points = torch.tensor(orthogonal_view_points).unsqueeze(dim=0).unsqueeze(dim=0)

    view_prediction = predictions["landmarks2d"][f"view_{view_id}"]
    view_P = torch.tensor(view_prediction["P_pfw"]).reshape(3, 4).unsqueeze(dim=0)

    # ts, ms = project_to_second_view_(view_P, orthogonal_view_P, view_points)

    ts, ms = project_to_second_view_fixed(orthogonal_view_P, view_P, orthogonal_view_points)
    ts = ts[0, 0].cpu().numpy()
    ms = ms[0, 0].cpu().numpy()

    # Create the first image (overlayed arrays)
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111)

    # Plot array1 in red and array2 in blue
    ax.imshow(view, cmap='gray')
    plt.tight_layout()

    for screw_i, (t, m) in enumerate(zip(ts, ms)):
        screw_i = screw_i // 2
        ax.plot([t[0], t[0] + m[0] * x_max], [t[1], t[1] + m[1] * x_max],
                 color=colors[screw_i], linestyle='--', lw=4)

        y_offset = 25

        vertices = [
            (0, t[1] + y_offset),
            (976, t[1] + m[1] * x_max + y_offset),
            (976, t[1] + m[1] * x_max - y_offset),
            (0, t[1] - y_offset)
        ]

        # Create a Polygon patch
        quadrilateral = Polygon(vertices, closed=True, linewidth=1, edgecolor=colors[screw_i],
                                facecolor=colors[screw_i], alpha=0.4)
        ax.add_patch(quadrilateral)

    plt.xlim(0, 976)
    plt.ylim(976, 0)

    plt.show()

    # Save just the portion _inside_ the second axis's boundaries
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"{dir}_{view_id}_custom.png", bbox_inches=extent)


if __name__ == "__main__":
    # verify_projection()
    projection_in_image(0, 180)
    projection_in_image(180, 0)
