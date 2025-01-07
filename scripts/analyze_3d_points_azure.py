import os
import json
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def get_camera_center(_P):
    return -np.linalg.inv(_P[:3, :3]) @ _P[:3, 3]


def compute_alpha_angle(P0, P1, pt2d):
    # construct 0-degree epipolar plane from camera positions and isocenter
    C0, C1, iso_center = get_camera_center(P0), get_camera_center(P1), np.zeros(3)

    e1 = P0 @ np.pad(C1, (0, 1), constant_values=1)
    e1 = e1[:2] / e1[2]

    # iso-center lies at (255.5, 255.5, 255.5)
    i1 = P0 @ np.array([0, 0, 0, 1])
    i1 = i1[:2] / i1[2]

    b = i1 - e1
    b /= np.linalg.norm(b)

    d = pt2d - e1
    d /= np.linalg.norm(d)

    # Calculate the angle using arctan2
    alpha2 = np.arctan2(b[1], b[0]) - np.arctan2(d[1], d[0])
    # Normalize the angle to be between -pi and pi
    alpha2 = (alpha2 + np.pi) % (2 * np.pi) - np.pi

    return np.rad2deg(alpha2)


def shortest_path_between_lines(P1, P2, P3, P4):
    P1, P2, P3, P4 = map(np.array, [P1, P2, P3, P4])
    d1 = (P2 - P1) / np.linalg.norm(P2 - P1)
    d2 = (P4 - P3) / np.linalg.norm(P4 - P3)
    n = np.cross(d1, d2)
    n_norm = np.linalg.norm(n)

    if n_norm == 0:
        raise ValueError("The lines are parallel and do not have a unique shortest path.")

    P13 = P3 - P1
    A, B, C = np.dot(d1, d1), np.dot(d1, d2), np.dot(d2, d2)
    D, E = np.dot(d1, P13), np.dot(d2, P13)

    denominator = A * C - B * B
    if denominator == 0:
        raise ValueError("The lines are parallel and do not have a unique shortest path.")

    t = (B * E - C * D) / denominator
    s = (A * E - B * D) / denominator

    closest_point_on_L1 = P1 - t * d1
    closest_point_on_L2 = P3 - s * d2

    return (closest_point_on_L1 + closest_point_on_L2) / 2


def line_to_line_distance(p1, p2, q1, q2):
    v1 = p2 - p1
    v2 = q2 - q1
    v12 = np.cross(v1, v2)
    v1_q1 = q1 - p1
    return np.abs(np.dot(v1_q1, v12)) / np.linalg.norm(v12)


def get_3d_points(points_2d, camera_matrix):
    batch_size = points_2d.shape[0]
    ones = np.ones((batch_size, 1))
    points_2d_homogeneous = np.hstack([points_2d, ones])
    pinv_camera_matrix = np.linalg.pinv(camera_matrix)
    points_3d_homogeneous = points_2d_homogeneous @ pinv_camera_matrix.T
    points_3d = points_3d_homogeneous[:, :3] / points_3d_homogeneous[:, 3, np.newaxis]
    return points_3d


def compute_3D_loc(p, p_aux, P_pfw, P_pfw_aux, detector_shape):
    P_pfw = np.array(P_pfw).reshape((3, 4))
    P_pfw_aux = np.array(P_pfw_aux).reshape((3, 4))

    p_heads = np.array([o["head"] for o in p.values() if o["screw_prob"] > 0], dtype=np.double) * detector_shape[0]
    p_tips = np.array([o["tip"] for o in p.values() if o["screw_prob"] > 0], dtype=np.double) * detector_shape[0]

    p_aux_heads = np.array([o["head"] for o in p_aux.values() if o["screw_prob"] > 0], dtype=np.double) * detector_shape[0]
    p_aux_tips = np.array([o["tip"] for o in p_aux.values() if o["screw_prob"] > 0], dtype=np.double) * detector_shape[0]

    if len(p_tips) == 0 or len(p_heads) == 0:
        return None, None

    if len(p_aux_tips) == 0 or len(p_aux_heads) == 0:
        return None, None

    p_tips_3d = get_3d_points(p_tips, P_pfw)
    p_heads_3d = get_3d_points(p_heads, P_pfw)

    p_aux_heads_3d = get_3d_points(p_aux_heads, P_pfw_aux)
    p_aux_tips_3d = get_3d_points(p_aux_tips, P_pfw_aux)

    c = -np.linalg.inv(P_pfw[:3, :3]) @ P_pfw[:3, 3]
    c_aux = -np.linalg.inv(P_pfw_aux[:3, :3]) @ P_pfw_aux[:3, 3]

    # Vectorized computation of distance matrices

    head_distances = np.empty((len(p_heads), len(p_aux_heads)))
    tip_distances = np.empty((len(p_tips), len(p_aux_tips)))

    for i in range(len(p_heads)):
        for j in range(len(p_aux_heads)):
            head_distances[i, j] = line_to_line_distance(p_heads_3d[i], c, p_aux_heads_3d[j], c_aux)

    for i in range(len(p_tips)):
        for j in range(len(p_aux_tips)):
            tip_distances[i, j] = line_to_line_distance(p_tips_3d[i], c, p_aux_tips_3d[j], c_aux)

    alpha_distances = np.empty((len(p_heads), len(p_aux_heads)))
    for i in range(len(p_heads)):
        for j in range(len(p_aux_heads)):
            alpha_distances[i, j] = np.abs(compute_alpha_angle(P_pfw, P_pfw_aux, p_heads[i]) -
                                     compute_alpha_angle(P_pfw_aux,P_pfw,p_aux_heads[j]))

    distance_matrix = head_distances + tip_distances + alpha_distances

    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    heads_3d = np.array(
        [shortest_path_between_lines(p_heads_3d[row_ind[i]], c, p_aux_heads_3d[col_ind[i]], c_aux) for i in range(min(row_ind.shape[0], col_ind.shape[0]))])
    tips_3d = np.array(
        [shortest_path_between_lines(p_tips_3d[row_ind[i]], c, p_aux_tips_3d[col_ind[i]], c_aux) for i in range(min(row_ind.shape[0], col_ind.shape[0]))])

    return heads_3d, tips_3d


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize predictions')

    if os.name == 'nt':
        image_dir_default = r"C:\Users\wagne\Desktop"
        prediction_dir_default = r"C:\Users\wagne\Desktop"
    else:
        # image_dir_default = r"/home/vault/iwi5/iwi5163h"
        # prediction_dir_default = r"/home/hpc/iwi5/iwi5163h/eff_detr_2024_06_10_02_52"
        image_dir_default = r"/home/vault/iwi5/iwi5165h"
        prediction_dir_default = r"/home/hpc/iwi5/iwi5165h/Screw-Detection-Results/Job-xxxxxx"

    parser.add_argument('--image_dir', type=str, default=image_dir_default,
                        help='Directory containing the images')
    parser.add_argument('--prediction_dir', type=str, default=prediction_dir_default,
                        help='Directory containing the predictions')

    args = parser.parse_args()

    datadir_b = Path(args.image_dir) / "V1-1to3objects-400projections-circular"
    datadir_p = Path(args.prediction_dir) / "V1-1to3objects-400projections-circular"
    detector_shape, pixel_size = (976, 976), 0.305

    # loop over folders and convert each set of tool volumes to landmarks
    ##############################################################################################
    # samples = [f for f in os.listdir(datadir_p) if os.path.isdir(datadir_p / f)] # Original code
    
    evaluation_set = ['Ankle21', 'Ankle23', 'Elbow04', 'Wrist11', 'Wrist12', 'Wrist13', 'Spine06', 'Spine07', 'Ankle19', 'Wrist08', 'Wrist09', 'Wrist10']
    samples = []
    for sample in evaluation_set:
        if os.path.isdir(datadir_p / (sample + "_le_512x512x512_1")):
            samples.append(sample + "_le_512x512x512_1")
        if os.path.isdir(datadir_p / (sample + "_le_512x512x512_2")):
            samples.append(sample + "_le_512x512x512_2")
        if os.path.isdir(datadir_p / (sample + "_le_512x512x512_3")):
            samples.append(sample + "_le_512x512x512_3")
    ##############################################################################################

    print(f"processing {len(samples)} samples:")

    overall_avg_loss_distance_head_mm = 0
    overall_avg_loss_distance_tip_mm = 0
    overall_avg_angle_deviation = 0
    overall_avg_cardinality = 0

    overall_screws = 0

    for s, sample in enumerate(samples):
        folder_b = datadir_b / sample
        folder_p = datadir_p / sample

        avg_loss_distance_head_mm = 0
        avg_loss_distance_tip_mm = 0
        avg_angle_deviation = 0
        avg_cardinality = 0

        total_screws = 0

        heads_list = []
        tips_list = []

        # debug plot
        # if (folder_b / "labels.json").exists() and (folder_p / "predictions_test_50.json").exists(): # Original code

        # if (folder_b / "labels.json").exists() and ((folder_p / "predictions_test_50.json").exists() or (folder_p / "predictions_val_49.json").exists()): # Adapted code
        if (folder_b / "labels.json").exists() and ((folder_p / "predictions_test_50.json").exists() or (folder_p / "predictions_val_48.json").exists()): # Adapted code

            # read projections
            print(f"\ncomputing 3D loss for '{folder_b}' ...")
            tgt = json.load(open(folder_b / "labels.json"))

            # pred = json.load(open(folder_p / "predictions_test_50.json")) # Original code

            # Adapted code
            if (folder_p / "predictions_test_50.json").exists():
                pred = json.load(open(folder_p / "predictions_test_50.json"))
            else:
                # pred = json.load(open(folder_p / "predictions_val_49.json"))
                pred = json.load(open(folder_p / "predictions_val_48.json"))

            views = []

            pred = pred['landmarks2d']
            pred_views = [p["predictions"] for p in pred.values()]
            tgt_views = [p["targets"] for p in pred.values()]

            # # testing
            # pred_views = [p["targets"] for p in pred.values()]
            # for pred_view in pred_views:
            #     for o in pred_view.values():
            #         o["screw_prob"] = 1
            # # testing

            # correct to y, x format
            for obj in pred_views:
                for key, value in obj.items():
                    value['head'][0], value['head'][1] = value['head'][1], value['head'][0]
                    value['tip'][0], value['tip'][1] = value['tip'][1], value['tip'][0]

            # P_pfws = np.stack([tgt["landmarks2d"][k]["P_pfw"] for k in pred.keys()], dtype=np.double)
            P_pfws = np.stack([tgt["landmarks2d"][k]["P_pfw"] for k in pred.keys()]) # Bug fix
            P_pfws = P_pfws.astype(np.double) # Bug fix

            tgt = tgt['landmarks3d'].values()
            tgt_heads = np.array([o[0] for o in tgt])
            tgt_tips = np.array([o[1] for o in tgt])

            diff = 9

            for i in range(len(pred_views) - diff):

                heads_list.append([])
                tips_list.append([])

                screw_outside = False
                for key, value in tgt_views[i].items():
                    if (value['head'][0] >= 1. or value['tip'][0] >= 1.) or (
                            value['head'][1] >= 1. or value['tip'][1] >= 1.) or (
                            value['head'][0] <= 0. or value['tip'][0] <= 0.) or (
                            value['head'][1] <= 0. or value['tip'][1] <= 0.):
                        screw_outside = True
                for key, value in tgt_views[i + diff].items():
                    if (value['head'][0] >= 1. or value['tip'][0] >= 1.) or (
                            value['head'][1] >= 1. or value['tip'][1] >= 1.) or (
                            value['head'][0] <= 0. or value['tip'][0] <= 0.) or (
                            value['head'][1] <= 0. or value['tip'][1] <= 0.):
                        screw_outside = True
                if (screw_outside):
                    continue

                pred_heads, pred_tips = compute_3D_loc(pred_views[i], pred_views[i + diff], P_pfws[i], P_pfws[i + diff],
                                                       detector_shape)

                if (pred_heads is None) or (pred_tips is None):
                    continue

                print(f"view {i} and {i + diff}")

                for num_screws in range(min(len(pred_heads), len(pred_tips))):
                    heads_list[i].append(list(pred_heads[num_screws]))
                    tips_list[i].append(list(pred_tips[num_screws]))

                # Calculate the pairwise distance matrices using NumPy
                distance_matrix_head = np.linalg.norm(pred_heads[:, np.newaxis] - tgt_heads, axis=2)
                distance_matrix_tip = np.linalg.norm(pred_tips[:, np.newaxis] - tgt_tips, axis=2)

                # Use the linear_sum_assignment function from SciPy to find the optimal assignment
                row_ind, col_ind = linear_sum_assignment(distance_matrix_head + distance_matrix_tip)

                # Calculate the average loss distances
                avg_loss_distance_head_mm += np.sum(distance_matrix_head[row_ind, col_ind])
                avg_loss_distance_tip_mm += np.sum(distance_matrix_tip[row_ind, col_ind])

                pred_vectors = pred_tips - pred_heads
                pred_vectors /= np.linalg.norm(pred_vectors, axis=1)[:, np.newaxis]
                tgt_vectors = tgt_tips - tgt_heads
                tgt_vectors /= np.linalg.norm(tgt_vectors, axis=1)[:, np.newaxis]

                # Calculate dot products and magnitudes
                cos_angles = np.einsum('ij,ij->i', pred_vectors[row_ind], tgt_vectors[col_ind])
                angles_radians = np.arccos(np.clip(cos_angles, -1.0, 1.0))  # Clip to handle numerical errors

                # Convert angles to degrees
                avg_angle_deviation += np.sum(np.degrees(angles_radians))

                avg_cardinality = np.abs(len(row_ind) - len(col_ind))

                # Calculate the total number of screws
                total_screws += int(len(row_ind))

            if total_screws == 0:
                print("no screws fully inside")
                continue

            overall_avg_loss_distance_head_mm += avg_loss_distance_head_mm
            overall_avg_loss_distance_tip_mm += avg_loss_distance_tip_mm
            overall_avg_angle_deviation += avg_angle_deviation
            overall_screws += total_screws

            avg_loss_distance_head_mm = (avg_loss_distance_head_mm / total_screws)
            avg_loss_distance_tip_mm = (avg_loss_distance_tip_mm / total_screws)
            avg_angle_deviation /= total_screws

            overall_avg_cardinality += avg_cardinality
            avg_cardinality /= total_screws

            print(f"avg loss head: {avg_loss_distance_head_mm:.3f}mm")
            print(f"avg loss tip: {avg_loss_distance_tip_mm:.3f}mm")
            print(f"avg angle deviation: {avg_angle_deviation:.3f}°")
            print(f"avg cardinality deviation: {int(avg_cardinality)}")

        print(f"heads_list = {heads_list}")
        print(f"tips_list = {tips_list}")

    print(f"--------------------------------------------------------------------------------------")
    print(f"full test dataset avg loss head: {overall_avg_loss_distance_head_mm / overall_screws:.3f}mm")
    print(f"full test dataset avg loss tip: {overall_avg_loss_distance_tip_mm / overall_screws:.3f}mm")
    print(f"full test dataset avg angle deviation: {overall_avg_angle_deviation / overall_screws:.3f}°")
    print(f"full test dataset avg cardinality deviation: {int(overall_avg_cardinality / overall_screws)}")
    print(f"done ({s+1}/{len(samples)})\n")
