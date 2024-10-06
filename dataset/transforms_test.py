import unittest
import torch

from dataset.transform_information import PointTransformation
from dataset.transforms import get_transformation_matrix, apply_transformation_matrix_batched, \
    reverse_transformation_matrix_batched


class TestTransformationComparison(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_points = torch.rand(1, 100, 2, device=self.device)

    def test_transformation_comparison(self):
        test_cases = [
            (True, False, 0, (0, 0, 976, 976), 976),
            (False, True, 90, (100, 1, 776, 776), 1024),
            (True, True, 180, (50, 600, 876, 876), 976),
            (False, False, 270, (0, 0, 976, 976), 1024),
        ]

        for h_flip, v_flip, angle, crop_region, orig_size in test_cases:
            with self.subTest(h_flip=h_flip, v_flip=v_flip, angle=angle, crop_region=crop_region, orig_size=orig_size):
                # Original implementation
                matrix = get_transformation_matrix(h_flip, v_flip, angle, crop_region, orig_size, self.device)
                original_projected = apply_transformation_matrix_batched(self.random_points, matrix)
                original_reprojected = reverse_transformation_matrix_batched(original_projected.unsqueeze(2), matrix)

                # New implementation
                point_transform = PointTransformation(h_flip, v_flip, angle, crop_region, orig_size)
                new_projected = point_transform.project(self.random_points.squeeze(0))
                new_reprojected = point_transform.reproject(new_projected)

                # Compare results
                self.assertTrue(torch.allclose(original_projected.squeeze(), new_projected, atol=1e-3),
                                "Projection mismatch")
                self.assertTrue(torch.allclose(original_reprojected.squeeze(), new_reprojected, atol=1e-3),
                                "Reprojection mismatch")

    def test_point_transformation_validation(self):
        with self.assertRaises(ValueError):
            PointTransformation(angle=45)

        with self.assertRaises(ValueError):
            PointTransformation(crop_region=(0, 0, 976))

        with self.assertRaises(ValueError):
            PointTransformation(crop_region=(0, 0, 976, "976"))

        with self.assertRaises(ValueError):
            PointTransformation(orig_size=0)

    def test_edge_cases(self):
        extreme_points = torch.tensor([[[0, 0], [1, 1], [0.5, 0.5]]], device=self.device)

        transform = PointTransformation(h_flip=True, v_flip=True, angle=180, crop_region=(100, 100, 776, 776),
                                        orig_size=1024)

        projected = transform.project(extreme_points)
        reprojected = transform.reproject(projected)

        self.assertTrue(torch.allclose(extreme_points, reprojected, atol=1e-5),
                        "Edge case: Extreme points don't match after projection and reprojection")

        matrix = get_transformation_matrix(h_flip=True, v_flip=True, angle=180, crop_region=(100, 100, 776, 776),
                                           orig_size=1024, device=self.device)
        projected = apply_transformation_matrix_batched(extreme_points, matrix)
        reprojected = reverse_transformation_matrix_batched(projected.unsqueeze(1), matrix)

        self.assertTrue(torch.allclose(extreme_points, reprojected, atol=1e-5),
                        "Edge case: Extreme points don't match after projection and reprojection")

    def test_batch_processing(self):
        batch_sizes = [1, 10, 100]
        num_points_list = [1, 10, 1000]

        for batch_size in batch_sizes:
            for num_points in num_points_list:
                with self.subTest(batch_size=batch_size, num_points=num_points):
                    points = torch.rand(batch_size, num_points, 2, device=self.device)

                    transform = PointTransformation(h_flip=True, v_flip=False, angle=90, crop_region=(50, 50, 876, 876),
                                                    orig_size=1024)

                    projected = transform.project(points)
                    reprojected = transform.reproject(projected)

                    self.assertEqual(projected.shape, points.shape, "Projected shape mismatch")
                    self.assertEqual(reprojected.shape, points.shape, "Reprojected shape mismatch")
                    self.assertTrue(torch.allclose(points, reprojected, atol=1e-5), "Batch processing failed")


if __name__ == "__main__":
    unittest.main()
