import json
import os

import numpy as np
from tqdm import tqdm


def generate_labels_for_volumes(data_dir, volume_names, images_per_volume):
    print(f"Started Generating Labels ...")

    label_objects_map = {}
    p_pfw_map = {}

    for volume_name in tqdm(volume_names):
        volume_path = os.path.join(data_dir, volume_name)
        labels_path = os.path.join(volume_path, "labels.json")

        objects_list, p_pfw_list = generate_labels_batched(labels_path, images_per_volume)

        label_objects_map[volume_name] = objects_list
        p_pfw_map[volume_name] = p_pfw_list

    return label_objects_map, p_pfw_map


def generate_labels_batched(labels_path, images_per_volume):
    with open(labels_path) as labels_file:
        labels = json.load(labels_file)

        objects_list = []
        p_pfw_list = []
        for view_idx in range(images_per_volume):
            objects, p_pfw = generate_labels(labels, view_idx)

            objects_list.append(objects)
            p_pfw_list.append(p_pfw)

    return objects_list, p_pfw_list


def generate_labels(labels, view_idx):
    # objects = labels["landmarks2d"][f"view_{view_idx}"]["objects"]
    # p_pfw = np.array(labels["landmarks2d"][f"view_{view_idx}"]["P_pfw"]).reshape((3, 4))

    objects = None
    p_pfw = None

    return objects, p_pfw


def get_meta_data(data_dir, volume_name):
    # volume_path = os.path.join(data_dir, volume_name)
    # labels_path = os.path.join(volume_path, "labels.json")

    # with open(labels_path) as labels_file:
    #     labels = json.load(labels_file)
    #     detector_shape = labels["landmarks2d"][f"view_0"]["detector_shape"]
    #     pixel_size = labels["landmarks2d"][f"view_0"]["pixel_size"]

    detector_shape = None,
    pixel_size = None

    return detector_shape, pixel_size






