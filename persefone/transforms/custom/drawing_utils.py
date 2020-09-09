import math
import random
from typing import Tuple, Sequence

import cv2
import numpy as np
from scipy.interpolate import interp1d
from skimage.draw import ellipse_perimeter


class DrawingUtils:

    @classmethod
    def apply_patch(cls,
                    img: np.ndarray,
                    img_mask: np.ndarray,
                    corruption: np.ndarray,
                    corruption_mask: np.ndarray,
                    pos: Tuple[int, int]) -> None:
        """Applies a patch on a target image

        :param img: target image
        :type img: np.ndarray
        :param img_mask: target image mask
        :type img_mask: np.ndarray
        :param corruption: patch to apply
        :type corruption: np.ndarray
        :param corruption_mask: patch mask
        :type corruption_mask: np.ndarray
        :param pos: upper left patch corner in img coords
        :type pos: Tuple[int, int]
        """
        size_x = max(0, min(img.shape[0], pos[0] + corruption.shape[0]) - max(0, pos[0]))
        size_y = max(0, min(img.shape[1], pos[1] + corruption.shape[1]) - max(0, pos[1]))

        ra = max(0, pos[0])
        ca = max(0, pos[1])
        rb = ra + size_x
        cb = ca + size_y

        ra_p = -pos[0] if pos[0] < 0 else 0
        ca_p = -pos[1] if pos[1] < 0 else 0
        rb_p = ra_p + size_x
        cb_p = ca_p + size_y

        corruption = corruption[ra_p: rb_p, ca_p: cb_p]
        corruption_mask = corruption_mask[ra_p: rb_p, ca_p: cb_p]
        corruption_mask_3 = np.stack([corruption_mask] * 3, axis=-1)
        img[ra: rb, ca: cb, :] = img[ra: rb, ca: cb, :] * (1 - corruption_mask_3) + corruption * corruption_mask_3
        if img_mask is not None:
            img_mask[ra: rb, ca: cb] = np.maximum(img_mask[ra: rb, ca: cb], corruption_mask)

    @classmethod
    def ellipse(cls, size: Tuple[int, int], angle: float = 0., max_points: int = -1) -> Sequence[Tuple[int, int]]:
        """Samples points from the perimeter of an ellipse

        :param size: the radii of the ellipse
        :type size: Tuple[int, int]
        :param angle: ellipse rotation angle, defaults to 0.
        :type angle: float, optional
        :param max_points: the maximum number of points to sample, defaults to -1
        :type max_points: int, optional
        :return: a sequence of sampled points
        :rtype: Sequence[Tuple[int, int]]
        """
        radii = (math.floor((size[1] - 1) / 2), math.floor((size[0] - 1) / 2))
        rr, cc = ellipse_perimeter(radii[0], radii[1], radii[0], radii[1], angle)
        points = [(r, c) for r, c in zip(rr, cc)]
        points = sorted(points, key=lambda p: DrawingUtils.clockwise_angle(p, radii))
        if max_points > 0:
            points = points[::len(points) // max_points]
        return points

    @classmethod
    def polygon(cls,
                points: Sequence[Tuple[int, int]],
                color: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Draws a filled polygon on a numpy array

        :param points: vertex sequence
        :type points: Sequence[Tuple[int, int]]
        :param color: fill color
        :type color: Tuple[int, int, int]
        :return: image tensor and mask
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        min_x = min(p[0] for p in points)
        min_y = min(p[1] for p in points)
        size = (max(p[0] for p in points) - min_x,
                max(p[1] for p in points) - min_y)
        points = tuple((p[1] - min_y, p[0] - min_x) for p in points)
        mask = np.zeros(size)
        points = np.array(points, np.int32)
        cv2.drawContours(mask, [points], -1, 1.0, -1)
        corruption = np.stack([mask * color[i] for i in range(3)], axis=-1)
        return corruption, mask

    @classmethod
    def interpolate(cls, points: Sequence[Tuple[int, int]], overlap: int = 3) -> Sequence[Tuple[int, int]]:
        """Performs a cubic interpolation of a sequence of points

        :param points: the sequence of points to interpolate
        :type points: Sequence[Tuple[int, int]]
        :param overlap: overlap, defaults to 3
        :type overlap: int, optional
        :return: a sequence of points obtained from cubic interpolation
        :rtype: Sequence[Tuple[int, int]]
        """
        indexes = np.arange(0, len(points) + 2 * overlap)
        interp_i = np.linspace(overlap, len(points) + overlap, 10 * len(points))
        points = points[-overlap:] + points + points[:overlap]
        xi = interp1d(indexes, np.array(points)[:, 0], kind='cubic')(interp_i)
        yi = interp1d(indexes, np.array(points)[:, 1], kind='cubic')(interp_i)
        res = [(int(i), int(j)) for i, j in zip(xi, yi)]
        return res

    @classmethod
    def random_displace(cls, points: Sequence[Tuple[int, int]], radius: float) -> Sequence[Tuple[int, int]]:
        """Randomly displaces points given a radius

        :param points: a sequence of points to displace
        :type points: Sequence[Tuple[int, int]]
        :param radius: the displacement radius
        :type radius: float
        :return: the displaced sequence of points
        :rtype: Sequence[Tuple[int, int]]
        """
        res = []
        for point in points:
            norm = random.random() * radius
            angle = random.random() * 2 * math.pi
            dx, dy = int(norm * math.cos(angle)), int(norm * math.sin(angle))
            res.append((point[0] + dy, point[1] + dx))
        return res

    @classmethod
    def clockwise_angle(cls, point: Tuple[int, int], center: Tuple[int, int]) -> float:
        """Returns the angle between a 2d point and the vector (0, 1) wrt a center point c

        :param point: the point
        :type point: Tuple[int, int]
        :param center: the center point
        :type center: Tuple[int, int]
        :return: the angle in radians
        :rtype: float
        """
        refvec = [0, 1]
        vector = [point[0] - center[0], point[1] - center[1]]
        norm = math.hypot(vector[0], vector[1])
        if norm == 0:
            return -math.pi
        normalized = [vector[0] / norm, vector[1] / norm]
        dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]
        diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]
        angle = math.atan2(diffprod, dotprod)
        if angle < 0:
            return 2 * math.pi + angle
        return angle
