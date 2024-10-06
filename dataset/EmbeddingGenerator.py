import os

import numpy as np
import tifffile
import torch
from tifffile import imwrite
from tqdm import tqdm

from utils.data_normalization import neglog_normalize


def generate_embedding_for_volumes(data_dir, volumes, backbone, images_per_volume):

    if backbone.need_projection:
        print(f"Nothing TODO ...")
        return

    print(f"Started Generating Embeddings ...")

    backbone.instantiate()

    for volume in tqdm(volumes):
        volume_path = os.path.join(data_dir, volume)

        embeddings = generate_embedding_batched(backbone, images_per_volume, volume_path)

        # store embeddings batched
        embeddings_path = os.path.join(volume_path, "embeddings.tiff")
        if isinstance(embeddings[0], list):
            for i in range(len(embeddings[0])):
                embedding_list = [embedding[i] for embedding in embeddings]
                imwrite(embeddings_path.replace(".tiff", f"{i}.tiff"), embedding_list, dtype=np.float32, bigtiff=True)
        else:
            imwrite(embeddings_path, embeddings, dtype=np.float32)

    print(f"Generated Embeddings for: {volumes}")


def generate_embedding_batched(backbone, images_per_volume, volume_path):

    projections_path = os.path.join(volume_path, "projections.tiff")
    embeddings = []

    for view_idx in range(images_per_volume):
        embedding = generate_embedding(backbone, projections_path, view_idx)

        # Add the embedding to list
        if isinstance(embedding, list):
            embeddings.append(embedding)
        else:
            embeddings.append(embedding.squeeze())

    return embeddings


def generate_embedding(backbone, projections_path, view_idx):
    # load encoding
    with tifffile.TiffFile(projections_path) as projection_file:
        # Access the specific tensor (image) by index
        projection = projection_file.asarray(key=slice(view_idx, (view_idx + 1)))
        projection = neglog_normalize(projection)

    return backbone.forward(projection)


def load_embeddings_from_tmp(view_idx, volume_path, feature_dim):
    embeddings_path = os.path.join(volume_path, "embeddings.tiff")
    with tifffile.TiffFile(embeddings_path) as embeddings:
        # Access the specific tensor (image) by index
        embedding = embeddings.asarray(key=slice(view_idx * feature_dim, (view_idx + 1) * feature_dim))
        embedding_tensor = torch.from_numpy(embedding)
    return embedding_tensor
