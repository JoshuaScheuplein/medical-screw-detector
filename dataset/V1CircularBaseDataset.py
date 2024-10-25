import json
import os

import tifffile
import torch
from torch.utils.data import Dataset

from dataset.LabelGenerator import generate_labels_for_volumes, generate_labels, get_meta_data
from utils.data_normalization import neglog_normalize


class V1CircularBaseDataset(Dataset):
    def __init__(self, data_dir, volume_names, images_per_volume, neglog_normalize, transform, pre_load_labels):
        self.data_dir = data_dir
        self.volume_names = volume_names
        self.images_per_volume = images_per_volume
        self.neglog_normalize = neglog_normalize

        self.transform = transform

        self.pre_load_labels = pre_load_labels

        if self.pre_load_labels:
            label_objects_map, label_p_pfw_map = generate_labels_for_volumes(data_dir, volume_names, images_per_volume)
            self.label_objects_map = label_objects_map
            self.label_P_pfw_map = label_p_pfw_map

        detector_shape, pixel_size = get_meta_data(data_dir, volume_names[0])
        self.detector_shape = detector_shape
        self.pixel_size = pixel_size

    def __len__(self):
        return len(self.volume_names) * self.images_per_volume

    def __getitem__(self, idx):
        volume_idx = idx // self.images_per_volume
        view_idx = idx % self.images_per_volume

        volume_path = os.path.join(self.data_dir, self.volume_names[volume_idx])

        projections_path = os.path.join(volume_path, "projections.tiff")
        with tifffile.TiffFile(projections_path) as projection_file:
            projection = projection_file.asarray(key=slice(view_idx, (view_idx + 1)))
            if neglog_normalize:
                projection = neglog_normalize(projection)

        projection_tensor = torch.from_numpy(projection)
        print("Shape", projection_tensor.shape)
        assert projection_tensor.shape == (400, 976, 976) # Additionally added

        # load objects
        if self.pre_load_labels:
            objects = self.label_objects_map[self.volume_names[volume_idx]][view_idx]
            p_pfw = self.label_P_pfw_map[self.volume_names[volume_idx]][view_idx]
        else:
            labels_path = os.path.join(volume_path, "labels.json")

            with open(labels_path) as labels_file:
                labels = json.load(labels_file)
                objects, p_pfw = generate_labels(labels, view_idx)

        return projection_tensor, (objects, p_pfw)
