import torch
from matplotlib import pyplot as plt

from dataset.Objects import ScrewEnum
from dataset.V1CircularBaseDataset import V1CircularBaseDataset


class V1CircularScrewDataset(V1CircularBaseDataset):
    def __init__(self, data_dir, volume_names, images_per_volume, neglog_normalize, transform=None,
                 pre_load_labels=True, aux_view=False, visualize=True, reduction_factor=1):
        self.aux_view = aux_view
        self.visualize = visualize
        self.reduction_factor = reduction_factor
        super().__init__(data_dir, volume_names, images_per_volume, neglog_normalize, transform, pre_load_labels)

    def __len__(self):
        return super().__len__() // self.reduction_factor

    def __getitem__(self, idx):
        idx *= self.reduction_factor
        # Get projection and target for idx
        projection, (objects, p_pfw) = super().__getitem__(idx)
        target = self.generate_target(objects, p_pfw)
        projection, target = self.transform(projection, target)

        print(f"Item idx: {idx}")

        if self.aux_view:
            # Get projection and target for orthogonal view
            volume_idx = idx // self.images_per_volume
            view_idx = idx % self.images_per_volume
            if view_idx < self.images_per_volume // 2:
                orthogonal_view_idx = (view_idx + 180) + volume_idx * self.images_per_volume
            else:
                orthogonal_view_idx = (view_idx - 180) + volume_idx * self.images_per_volume

            projection_orthogonal, (objects_orthogonal, p_pfw_orthogonal) = super().__getitem__(orthogonal_view_idx)
            target_orthogonal = self.generate_target(objects_orthogonal, p_pfw_orthogonal)
            projection_orthogonal, target_orthogonal = self.transform(projection_orthogonal, target_orthogonal)

        if self.visualize and self.aux_view:
            print(f"idx: {idx}, orthogonal_view_idx: {orthogonal_view_idx}")

        # Visualize for verification
        if self.visualize:
            visualize(projection, target)
        if self.visualize and self.aux_view:
            visualize(projection_orthogonal, target_orthogonal)

        if self.aux_view:
            return [projection, projection_orthogonal], [target, target_orthogonal], idx
        else:
            return projection, target, idx

    def generate_target(self, objects, p_pfw):
        screws = []
        for screw in objects.values():
            screws.append([coord for coords in screw for coord in coords[::-1]])

        screws_tensor = torch.tensor(screws)
        landmark_screws = screws_tensor / self.detector_shape[0]
        landmark_screws = project_outlier_coordinates(landmark_screws)

        landmark_labels = torch.tensor(len(screws) * [ScrewEnum.SCREW], dtype=torch.int64)

        target = {
            "screws": landmark_screws,
            "labels": landmark_labels,
            "p_pfw": torch.from_numpy(p_pfw)
        }

        return target


def visualize(projection, target):
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(projection)
    for screw in target['screws']:
        head_x = screw[0] * projection.shape[1]
        head_y = screw[1] * projection.shape[1]

        tip_x = screw[2] * projection.shape[1]
        tip_y = screw[3] * projection.shape[1]

        plt.plot([head_x], [head_y], 'o', c='white')
        plt.plot([tip_x], [tip_y], 'x', c='white')
        plt.plot([head_x, tip_x], [head_y, tip_y], '--', c='white')
    fig.show()


def project_outlier_coordinates(tensor):
    def find_intersection(x1, y1, x2, y2):
        # Check if the line is vertical
        if x1 == x2:
            return x1, max(0, min(1, y1))

        # Calculate slope and y-intercept
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        # Check intersections with boundaries
        intersections = []

        # Left boundary (x = 0)
        y = b
        if 0 <= y <= 1:
            intersections.append((0, y))

        # Right boundary (x = 1)
        y = m + b
        if 0 <= y <= 1:
            intersections.append((1, y))

        # Bottom boundary (y = 0)
        if m != 0:
            x = -b / m
            if 0 <= x <= 1:
                intersections.append((x, 0))

        # Top boundary (y = 1)
        if m != 0:
            x = (1 - b) / m
            if 0 <= x <= 1:
                intersections.append((x, 1))

        # Find the intersection point closest to the original point
        if intersections:
            distances = [(x - x2) ** 2 + (y - y2) ** 2 for x, y in intersections]
            return intersections[distances.index(min(distances))]

        # If no intersection found (shouldn't happen in normal cases)
        return x2, y2

    normalized_tensor = tensor.clone()
    for i in range(tensor.shape[0]):
        x1, y1, x2, y2 = tensor[i]

        # Check if any point is outside [0, 1]
        if not (0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1):
            # If first point is outside, find intersection
            if not (0 <= x1 <= 1 and 0 <= y1 <= 1):
                x1, y1 = find_intersection(x2, y2, x1, y1)

            # If second point is outside, find intersection
            if not (0 <= x2 <= 1 and 0 <= y2 <= 1):
                x2, y2 = find_intersection(x1, y1, x2, y2)

            normalized_tensor[i] = torch.tensor([x1, y1, x2, y2])

    return normalized_tensor

