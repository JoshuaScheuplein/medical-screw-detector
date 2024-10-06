import unittest
import torch

from dataset.transform_information import project_points, reproject_points


class TestPointProjection(unittest.TestCase):

    def setUp(self):
        self.points = torch.tensor([[0.5, 0.5], [0.25, 0.75], [0.75, 0.25]])
        self.orig_size = 976

    def assert_close(self, a, b):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)

    def run_projection_test(self, h_flip, v_flip, angle, crop_region):
        points = self.points.clone()
        projected = project_points(points, h_flip, v_flip, angle, crop_region, self.orig_size)
        reprojected = reproject_points(projected, h_flip, v_flip, angle, crop_region, self.orig_size)
        self.assert_close(points, reprojected)

    # Original transformation tests
    def test_no_transform(self):
        self.run_projection_test(False, False, 0, (0, 0, self.orig_size, 1))

    def test_h_flip(self):
        self.run_projection_test(True, False, 0, (0, 0, self.orig_size, 1))

    def test_v_flip(self):
        self.run_projection_test(False, True, 0, (0, 0, self.orig_size, 1))

    def test_90_degree_rotation(self):
        self.run_projection_test(False, False, 90, (0, 0, self.orig_size, 1))

    def test_180_degree_rotation(self):
        self.run_projection_test(False, False, 180, (0, 0, self.orig_size, 1))

    def test_270_degree_rotation(self):
        self.run_projection_test(False, False, 270, (0, 0, self.orig_size, 1))

    def test_combined_transformations(self):
        self.run_projection_test(True, True, 90, (0, 0, self.orig_size, 1))

    # New crop region tests
    def test_center_crop(self):
        crop_size = 488  # Half of the original size
        start = (self.orig_size - crop_size) // 2
        self.run_projection_test(False, False, 0, (start, start, crop_size, 1))

    def test_top_left_crop(self):
        crop_size = 488
        self.run_projection_test(False, False, 0, (0, 0, crop_size, 1))

    def test_bottom_right_crop(self):
        crop_size = 488
        start = self.orig_size - crop_size
        self.run_projection_test(False, False, 0, (start, start, crop_size, 1))

    def test_vertical_strip_crop(self):
        crop_width = 244  # Quarter of the original size
        self.run_projection_test(False, False, 0, (0, (self.orig_size - crop_width) // 2, crop_width, 1))

    def test_horizontal_strip_crop(self):
        crop_height = 244
        self.run_projection_test(False, False, 0, ((self.orig_size - crop_height) // 2, 0, self.orig_size, 1))

    def test_small_center_crop(self):
        crop_size = 122  # Eighth of the original size
        start = (self.orig_size - crop_size) // 2
        self.run_projection_test(False, False, 0, (start, start, crop_size, 1))

    def test_large_center_crop(self):
        crop_size = 854  # Seven-eighths of the original size
        start = (self.orig_size - crop_size) // 2
        self.run_projection_test(False, False, 0, (start, start, crop_size, 1))

    def test_odd_sized_crop(self):
        crop_size = 501  # An odd number
        start_i = (self.orig_size - crop_size) // 2
        start_j = (self.orig_size - crop_size) // 2 + 1  # Offset by 1
        self.run_projection_test(False, False, 0, (start_i, start_j, crop_size, 1))

    def test_crop_with_transformations(self):
        crop_size = 488
        start = (self.orig_size - crop_size) // 2
        self.run_projection_test(True, True, 90, (start, start, crop_size, 1))

if __name__ == '__main__':
    unittest.main()
