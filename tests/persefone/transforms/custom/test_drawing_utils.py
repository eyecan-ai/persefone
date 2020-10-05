import math
import random
from typing import Tuple
import numpy as np
import pytest

from persefone.transforms.custom.drawing_utils import DrawingUtils


class TestDrawingUtils:

    @pytest.mark.parametrize(('point', 'center', 'expected'), (
        (
            (6, 5),
            (5, 5),
            math.pi / 2,
        ),
        (
            (6, 6),
            (5, 5),
            math.pi / 4,
        ),
        (
            (5, 6),
            (5, 5),
            0.,
        ),
        (
            (-6, 3),
            (-5, 3),
            3 * math.pi / 2,
        ),
    ),)
    def test_clockwise_angle(self, point: Tuple[int, int], center: Tuple[int, int], expected: float):

        result = DrawingUtils.clockwise_angle(point, center)
        assert math.isclose(result, expected)

    @pytest.mark.parametrize(('points', 'radius'), (
        ([tuple(random.randint(0, 100) for _ in range(2)) for _ in range(20)],
         random.randint(0, 10)) for _ in range(100)
    ),)
    def test_random_displace(self, points, radius):
        displaced = DrawingUtils.random_displace(points, radius)
        for p, d in zip(points, displaced):
            assert math.hypot(p[0] - d[0], p[1] - d[1]) <= radius

    @pytest.mark.parametrize(('points', 'samples'), (
        ([tuple(random.randint(0, 100) for _ in range(2)) for _ in range(20)],
         random.randint(0, 10)) for _ in range(100)
    ),)
    def test_interpolate(self, points, samples):
        interpolated = DrawingUtils.interpolate(points, samples)
        assert len(interpolated) == samples
        for p in interpolated:
            assert len(p) == 2
            assert isinstance(p[0], int)
            assert isinstance(p[1], int)

    @pytest.mark.parametrize(('points', 'color', 'expected_mask'), (
        (
            [(0, 0), (5, 0), (5, 5), (0, 5)],
            (128, 255, 0),
            np.ones((5, 5))
        ),
        (
            [(0, 0), (2, 0), (2, 5), (0, 5)],
            (128, 255, 0),
            np.ones((2, 5))
        ),
    ))
    def test_polygon(self, points, color, expected_mask):
        poly, mask = DrawingUtils.polygon(points, color)
        assert np.allclose(mask, expected_mask)
        expected_poly = np.stack([mask * color[i] for i in range(3)], axis=-1)
        assert np.allclose(poly, expected_poly)

    @pytest.mark.parametrize(('size', 'angle', 'max_points'), (
        (tuple(random.randint(1, 100) for _ in range(2)),
         random.uniform(0, 2 * math.pi),
         random.randint(10, 20)) for _ in range(100)
    ),)
    def test_ellipse(self, size, angle, max_points):
        points = DrawingUtils.ellipse(size, angle, max_points)
        assert len(points) <= max_points
        for p in points:
            assert len(p) == 2
            assert isinstance(p[0], int)
            assert isinstance(p[1], int)

    @pytest.mark.parametrize(('img', 'img_mask', 'patch', 'patch_mask', 'pos', 'expected', 'expected_mask'), (
        (
            np.transpose(np.array([[[0, 0, 40, 40],
                                    [200, 200, 200, 40],
                                    [200, 200, 200, 40],
                                    [0, 0, 40, 40]],
                                   [[0, 0, 40, 40],
                                    [10, 10, 10, 30],
                                    [10, 10, 10, 30],
                                    [0, 0, 40, 40]],
                                   [[100, 100, 100, 100],
                                    [200, 200, 200, 100],
                                    [200, 200, 200, 100],
                                    [100, 100, 100, 100]]]), (1, 2, 0)).astype('float'),
            None,
            np.transpose(np.array([[[50, 50],
                                    [10, 40]],
                                   [[0, 0],
                                    [20, 20]],
                                   [[30, 0],
                                    [0, 0]]]), (1, 2, 0)).astype('float'),
            np.ones((2, 2)),
            (1, 1),
            np.transpose(np.array([[[0, 0, 40, 40],
                                    [200, 50, 50, 40],
                                    [200, 10, 40, 40],
                                    [0, 0, 40, 40]],
                                   [[0, 0, 40, 40],
                                    [10, 0, 0, 30],
                                    [10, 20, 20, 30],
                                    [0, 0, 40, 40]],
                                   [[100, 100, 100, 100],
                                    [200, 30, 0, 100],
                                    [200, 0, 0, 100],
                                    [100, 100, 100, 100]]]), (1, 2, 0)).astype('float'),
            None,
        ),
    ),)
    def test_apply_patch(self, img, img_mask, patch, patch_mask, pos, expected, expected_mask):
        print(img.shape, patch.shape)
        DrawingUtils.apply_patch(img, img_mask, patch, patch_mask, pos)
        assert np.allclose(img, expected)
        if img_mask is not None:
            assert np.allclose(img_mask, expected_mask)

    @pytest.mark.parametrize(('size', 'ul', 'ur', 'll', 'lr', 'expected'), (
        (
            (3, 5),
            (100, 100, 100),
            (60, 60, 100),
            (0, 0, 0),
            (40, 40, 40),
            np.transpose(np.array([
                [[100, 90, 80, 70, 60],
                 [50, 50, 50, 50, 50],
                 [0, 10, 20, 30, 40]],
                [[100, 90, 80, 70, 60],
                 [50, 50, 50, 50, 50],
                 [0, 10, 20, 30, 40]],
                [[100, 100, 100, 100, 100],
                 [50, 55, 60, 65, 70],
                 [0, 10, 20, 30, 40]],
            ]), (1, 2, 0)).astype('float')
        ),
    ))
    def test_gradient(self, size, ul, ur, ll, lr, expected):
        res = DrawingUtils.gradient(size, ul, ur, ll, lr)
        assert np.allclose(res, expected)
