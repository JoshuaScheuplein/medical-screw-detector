import json
import os

import numpy as np
import tifffile
import torch

from tqdm import tqdm

from common.scripts.alpha import get_image_encodings


def generate_encoding_for_volumes(data_dir, volumes, images_per_volume, embeddings_sizes):
    print(f"Started Generating Encodings ...")

    encodings_map = {}

    for volume in tqdm(volumes):
        volume_path = os.path.join(data_dir, volume)

        encodings = generate_encoding_batched(images_per_volume, volume_path, embeddings_sizes)

        # store embeddings batched
        encodings_map[volume] = encodings
        # embeddings_path = os.path.join(volume_path, "embeddings.tiff")
        # imwrite(embeddings_path, embeddings, dtype=np.float32)

    return encodings_map


def generate_encoding_batched(images_per_volume, volume_path, embeddings_sizes):

    labels_path = os.path.join(volume_path, "labels.json")
    encodings_multilevel = []

    with open(labels_path) as labels_file:
        labels = json.load(labels_file)
        views = list(labels["landmarks2d"].values())

        assert len(views) >= images_per_volume

        for i in range(images_per_volume):
            # Generate image encodings
            matrix_pfv_1 = np.array(views[i]["P_pfw"]).reshape((3, 4))
            matrix_pfv_2 = np.array(views[(i + 180) % 360]["P_pfw"]).reshape((3, 4))

            encoding = generate_encoding(matrix_pfv_1, matrix_pfv_2, embeddings_sizes)
            encodings_multilevel.append(torch.from_numpy(encoding))

    return encodings_multilevel


def generate_encoding(matrix_pfv_1, matrix_pfv_2, embeddings_sizes):
    encodings = []
    for i, embeddings_size in enumerate(embeddings_sizes):
        encodings = get_image_encodings(matrix_pfv_1, matrix_pfv_2, image_size=1, embeddings_size=embeddings_size)
    return encodings


def load_encodings_from_tmp(view_idx, volume_path):
    encoding_path = os.path.join(volume_path, "encodings.tiff")
    with tifffile.TiffFile(encoding_path) as encodings:
        # Access the specific tensor (image) by index
        encoding = encodings.asarray(key=slice(view_idx, (view_idx + 1)))
        encoding_tensor = torch.from_numpy(encoding)
    return encoding_tensor




