import numpy as np
import torch

from dataset.Objects import HeadTipEnum
from dataset.V1CircularBaseDataset import V1CircularBaseDataset


class V1CircularHeadTipDataset(V1CircularBaseDataset):
    def __init__(self, data_dir, volume_names, images_per_volume, neglog_normalize, transform=None,
                 pre_load_labels=True):
        super().__init__(data_dir, volume_names, images_per_volume, neglog_normalize, transform, pre_load_labels)

    def __getitem__(self, idx):
        projection, (objects, p_pfw) = super().__getitem__(idx)

        heads, tips = np.array(list(zip(*objects.values())))

        landmark_coords = np.concatenate((heads, tips), axis=0, dtype=np.float32)
        landmark_coords = landmark_coords / self.detector_shape[0]

        landmark_labels = np.array(len(heads) * [HeadTipEnum.HEAD] + len(tips) * [HeadTipEnum.TIP], dtype=np.int64)

        landmark_logits = np.zeros((len(heads) + len(tips), len(list(HeadTipEnum))), dtype=np.float32)
        landmark_logits[np.arange(len(heads) + len(tips)), np.squeeze(landmark_labels)] = 1.

        target = {
            "coords": torch.from_numpy(landmark_coords),
            "labels": torch.from_numpy(landmark_labels)
        }

        return projection, target, idx
