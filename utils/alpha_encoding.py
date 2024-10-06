import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from tifffile import tifffile

from utils.data_normalization import neglog_normalize


def get_camera_center(_P):
    return -np.linalg.inv(_P[:3, :3]) @ _P[:3, 3]


def get_image_encodings_orig(P0, P1, embeddings_size, image_size=976):
    # construct 0-degree epipolar plane from camera positions and isocenter
    C0, C1, iso_center =  (P0), get_camera_center(P1), np.zeros(3)

    # Calculate the distance between points
    distance = image_size / embeddings_size

    # Adjust the start and end points
    start = distance / 2
    end = image_size - start

    # Create the grid
    xs = np.linspace(start, end, embeddings_size)
    ys = np.linspace(start, end, embeddings_size)

    encoding = np.zeros((embeddings_size, embeddings_size, 3), dtype=np.float32)

    e1 = P0 @ np.pad(C1, (0, 1), constant_values=1)
    e1 = e1[:2] / e1[2]

    # iso-center lies at (255.5, 255.5, 255.5)
    i1 = P0 @ np.array([0, 0, 0, 1])
    i1 = i1[:2] / i1[2]

    b = i1 - e1
    b /= np.linalg.norm(b)

    for x_idx, x in enumerate(xs):
        for y_idx, y in enumerate(ys):
            encoding[x_idx, y_idx, 0] = y
            encoding[x_idx, y_idx, 1] = x

            point2d = np.array([x, y])
            d = point2d - e1
            d /= np.linalg.norm(d)

            encoding[x_idx, y_idx, 2] = np.rad2deg(np.arccos(np.dot(d, b)))

    return encoding


def get_image_encodings_np(P0, P1, embeddings_size, image_size=976):
    # Construct 0-degree epipolar plane from camera positions and isocenter
    C0, C1 = get_camera_center(P0), get_camera_center(P1)

    # Calculate the distance between points
    distance = image_size / embeddings_size

    # Adjust the start and end points
    start = distance / 2
    end = image_size - start

    # Create the grid
    xs = np.linspace(start, end, embeddings_size)
    ys = np.linspace(start, end, embeddings_size)

    # Create meshgrid for x and y coordinates
    x_grid, y_grid = np.meshgrid(xs, ys)

    # Initialize encoding array
    encoding = np.zeros((embeddings_size, embeddings_size, 3), dtype=np.float32)

    # Assign x and y coordinates to encoding
    encoding[:, :, 0] = x_grid
    encoding[:, :, 1] = y_grid

    # Calculate e1
    e1 = P0 @ np.append(C1, 1)
    e1 = e1[:2] / e1[2]

    # Calculate i1
    i1 = P0 @ np.array([0, 0, 0, 1])
    i1 = i1[:2] / i1[2]

    # Calculate b
    b = i1 - e1
    b /= np.linalg.norm(b)

    # Calculate d
    d = np.stack((x_grid - e1[0], y_grid - e1[1]), axis=-1)
    d_norm = np.linalg.norm(d, axis=-1, keepdims=True)
    d /= d_norm

    # Calculate angle
    dot_product = np.sum(d * b, axis=-1)
    encoding[:, :, 2] = np.rad2deg(np.arccos(np.clip(dot_product, -1, 1))).transpose()

    return encoding


def get_image_encodings_torch(P0, P1, embeddings_size, crop_region, image_size=976):
    x_crop, y_crop, edge_crop, _ = crop_region

    # Construct 0-degree epipolar plane from camera positions and isocenter
    C0 = -torch.linalg.inv(P0[:3, :3]) @ P0[:3, 3]
    C1 = -torch.linalg.inv(P1[:3, :3]) @ P1[:3, 3]

    # Calculate the distance between points
    distance = image_size / embeddings_size

    # Adjust the start and end points
    midpoint_offset = ((1 / 2) * distance)
    start_x = x_crop + midpoint_offset
    start_y = y_crop + midpoint_offset
    end_x = x_crop + edge_crop - midpoint_offset
    end_y = y_crop + edge_crop - midpoint_offset

    # Create the grid
    xs = torch.linspace(start_x, end_x, embeddings_size, device=P0.device)
    ys = torch.linspace(start_y, end_y, embeddings_size, device=P0.device)

    # Create meshgrid for x and y coordinates
    x_grid, y_grid = torch.meshgrid(xs, ys, indexing='xy')

    # Initialize encoding tensor
    encoding = torch.zeros((embeddings_size, embeddings_size, 3), dtype=torch.float32, device=P0.device)

    # Assign x and y coordinates to encoding
    encoding[:, :, 0] = x_grid
    encoding[:, :, 1] = y_grid

    # Calculate e1
    e1 = P0 @ torch.cat([C1, torch.ones(1, device=P0.device)])
    e1 = e1[:2] / e1[2]

    # Calculate i1
    i1 = P0 @ torch.tensor([0, 0, 0, 1.0], device=P0.device, dtype=torch.double)
    i1 = i1[:2] / i1[2]

    # Calculate b
    b = i1 - e1
    b /= torch.norm(b)

    # Calculate d
    d = torch.stack((y_grid - e1[0], x_grid - e1[1]), dim=-1)
    d_norm = torch.norm(d, dim=-1, keepdim=True)
    d /= d_norm

    # Calculate angle
    dot_product = torch.sum(b * d, dim=-1)
    encoding[:, :, 2] = torch.rad2deg(torch.acos(torch.clamp(dot_product, -1, 1)))

    return encoding


def compute_point_from_angle(P0, P1, angle_deg):
    C0, C1 = get_camera_center(P0), get_camera_center(P1)

    # Calculate e1
    e1 = P0 @ np.append(C1, 1)
    e1 = e1[:2] / e1[2]

    # Calculate i1
    i1 = P0 @ np.array([0, 0, 0, 1])
    i1 = i1[:2] / i1[2]

    # Calculate b
    b = i1 - e1
    b /= np.linalg.norm(b)

    # Step 1: Convert angle from degrees to radians
    angle_rad = np.deg2rad(angle_deg)

    # Step 2: Compute the direction vector d using the angle and direction vector b
    # We need to find a vector d such that the angle between d and b is angle_rad
    # This can be done by rotating b by angle_rad around the origin
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Rotation matrix to rotate vector b by angle_rad
    rotation_matrix = np.array([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ])

    d = np.dot(rotation_matrix, b)

    # Step 3: Normalize the direction vector d
    d /= np.linalg.norm(d)

    # Step 4: Compute the coordinates (x, y) using the point e1 and direction vector d
    # Assuming we want a unit distance from e1 in the direction of d
    point2d = e1 + d

    return point2d

def compare_encodings(P0, P1, embeddings_size, image_size=976):
    # Run both versions
    start_time = time.time()
    original_encoding = get_image_encodings_np(P0, P1, embeddings_size, image_size)
    original_time = time.time() - start_time

    start_time = time.time()
    # torch_encoding = get_image_encodings(P0, P1, embeddings_size, image_size)
    torch_encoding = get_image_encodings_torch(torch.from_numpy(P0), torch.from_numpy(P1), embeddings_size, [0, 0, 976, 976], image_size)
    torch_encoding = torch_encoding.cpu().numpy()
    torch_encoding[:,:,2] = torch_encoding[:,:,2]
    torch_time = time.time() - start_time

    # Compare shapes
    shape_match = original_encoding.shape == torch_encoding.shape

    # Compare values
    max_diff = np.max(np.abs(original_encoding - torch_encoding))
    mean_diff = np.mean(np.abs(original_encoding - torch_encoding))

    # Check if the results are close enough (you may need to adjust the tolerance)
    tolerance = 1e-5
    is_close = np.allclose(original_encoding, torch_encoding, atol=tolerance)

    # Print results
    print(f"Embeddings size: {embeddings_size}")
    print(f"Shape match: {shape_match}")
    print(f"Maximum absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    print(f"Results are close (tolerance {tolerance}): {is_close}")
    print(f"Original version time: {original_time:.4f} seconds")
    print(f"Torch version time: {torch_time:.4f} seconds")
    print()


def run_comparison(embeddings_sizes):
    # Wrist04_3: View 219
    P0 = np.array([
        0.145533695,
        -0.7776426046999999,
        6.136282765999999,
        492.21468370089997,
        -5.908049073099999,
        -1.8428585843,
        0.008844449099999999,
        494.924102607,
        0.0002764918,
        -0.0015831658,
        -1.4744e-06,
        1.0
    ]).reshape(3, 4)

    # P0 = np.array([1.9329688512300716, 0.027870687771518573, -0.24733333471455365, 49.88670138064623,
    # 0.00641431618466804, 1.9535261982497045, -0.016222745937738217, -29.403051155572314,
    # 2.1432546971285593e-07, 6.962679174891631e-05, -0.000503167730183988, 1.1107149496126492]).reshape(3, 4)

    # Wrist04_3: View 399
    P1 = np.array([
        0.15277453625,
        -0.7761021454,
        6.136493392699999,
        492.56649455189995,
        -5.89124714405,
        -1.8974346114,
        0.0091038589,
        495.3296259615,
        0.0002905649,
        -0.0015800646,
        -1.5734e-06,
        1.0
    ]).reshape(3, 4)

    # P1 = np.array([1.9270121086164027, -0.2498278322436109, -0.031223506085813847, 78.91055669487207,
    # 0.008362597642717648, 0.060055542744486906, -1.9435212096813328, 965.2879649769776,
    # -1.8865523258570322e-06, -0.0004993069433854468, -8.124611888260149e-05, 1.1488133215287428]).reshape(3, 4)

    for size in embeddings_sizes:
        compare_encodings(P0, P1, size)


def verify():
    # Wrist04_3: View 219
    P0 = np.array([
        0.145533695,
        -0.7776426046999999,
        6.136282765999999,
        492.21468370089997,
        -5.908049073099999,
        -1.8428585843,
        0.008844449099999999,
        494.924102607,
        0.0002764918,
        -0.0015831658,
        -1.4744e-06,
        1.0
    ]).reshape(3, 4)

    # Wrist04_3: View 399
    P1 = np.array([
        0.7890835737499999,
        0.10704433185,
        6.159262276,
        496.62499899539995,
        1.6400124013500001,
        -5.99003523925,
        -0.0013500143999999993,
        510.9366276465001,
        0.0015936577,
        0.0002207231,
        1.7502e-06,
        1.0
    ]).reshape(3, 4)

    compute_point_from_angle(P0, P1, 6.1)

    encoding_219 = get_image_encodings_torch(torch.from_numpy(P0), torch.from_numpy(P1), 976, [0, 0, 976, 976])[:,:,2].cpu().numpy()
    encoding_399 = get_image_encodings_torch(torch.from_numpy(P1), torch.from_numpy(P0), 976, [0, 0, 976, 976])[:,:,2].cpu().numpy()

    with tifffile.TiffFile("E:\MA_Data\V1-1to3objects-400projections-circular\Elbow03_le_512x512x512_2\projections.tiff") as projection_file:
        view_219 = projection_file.asarray(key=slice(360, (360 + 1)))
        view_219 = neglog_normalize(view_219)

        view_399 = projection_file.asarray(key=slice(180, (180 + 1)))
        view_399 = neglog_normalize(view_399)

    # Create the first image (overlayed arrays)
    plt.figure(figsize=(12, 5))
    plt.subplot(121)

    # Plot array1 in red and array2 in blue
    plt.imshow(view_219, cmap='Reds', alpha=0.5)
    plt.imshow(encoding_219, cmap='Blues', alpha=0.5)

    # Create the second image (values in range (a, b) in red for array1)
    a, b = 5, 5.2  # Define the range (a, b)

    # Highlight values in range (a, b) for array1
    mask = (encoding_219 > a) & (encoding_219 < b)
    highlighted = np.ma.masked_where(~mask, encoding_219)
    plt.imshow(highlighted, cmap='hot_r', vmin=a, vmax=b)

    # Add colorbar
    cbar = plt.colorbar(label='Array Values')
    cbar.ax.set_ylabel('Array Values', rotation=270, labelpad=15)

    plt.subplot(122)

    # Plot array1 in red and array2 in blue
    plt.imshow(view_399, cmap='Reds', alpha=0.5)
    plt.imshow(encoding_399, cmap='Blues', alpha=0.5)

    # Highlight values in range (a, b) for array1
    mask = (encoding_399 > a) & (encoding_399 < b)
    highlighted = np.ma.masked_where(~mask, encoding_399)
    plt.imshow(highlighted, cmap='hot_r', vmin=a, vmax=b)

    # Add colorbar
    cbar = plt.colorbar(label='Array Values')
    cbar.ax.set_ylabel('Array Values', rotation=270, labelpad=15)

    plt.title('Overlayed Arrays with Highlighted Range')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # embeddings_sizes = [32, 64, 128, 256]
    # run_comparison(embeddings_sizes)

    verify()