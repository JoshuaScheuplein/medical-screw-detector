import numpy as np


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