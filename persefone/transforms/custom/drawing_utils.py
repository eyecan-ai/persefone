import math
import random
from typing import Iterable, Tuple, Sequence, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw
from albumentations.augmentations.transforms import Resize, ToFloat, PadIfNeeded
from scipy.interpolate import interp1d
from skimage.draw import ellipse_perimeter


class DrawingUtils:

    @classmethod
    def apply_patch(cls,
                    img: np.ndarray,
                    img_mask: np.ndarray,
                    patch: np.ndarray,
                    patch_mask: np.ndarray,
                    pos: Tuple[int, int],
                    blend_radius: int = None) -> None:
        """Applies a patch on a target image

        :param img: target image
        :type img: np.ndarray
        :param img_mask: target image mask
        :type img_mask: np.ndarray
        :param patch: patch to apply
        :type patch: np.ndarray
        :param patch_mask: patch mask
        :type patch_mask: np.ndarray
        :param pos: pos in img coords
        :type pos: Tuple[int, int]
        """
        if blend_radius is not None:
            r = blend_radius
            pad = PadIfNeeded(patch.shape[0] + 2 * r, patch.shape[1] + 2 * r, cv2.BORDER_CONSTANT)
            padded = pad(image=patch, mask=patch_mask)
            patch = padded['image']
            patch_mask = padded['mask']
            eroded = cv2.erode(patch_mask, np.ones((r, r), dtype='uint8'))
            blending = cv2.GaussianBlur(eroded * 255, (2 * r + 1, 2 * r + 1), 0)
            blending = ToFloat()(image=blending)['image'] * patch_mask
        else:
            blending = patch_mask
        blending = np.stack([blending] * 3, axis=-1)

        size_x = max(0, min(img.shape[0], pos[0] + patch.shape[0]) - max(0, pos[0]))
        size_y = max(0, min(img.shape[1], pos[1] + patch.shape[1]) - max(0, pos[1]))

        ra = max(0, pos[0])
        ca = max(0, pos[1])
        rb = ra + size_x
        cb = ca + size_y

        ra_p = -pos[0] if pos[0] < 0 else 0
        ca_p = -pos[1] if pos[1] < 0 else 0
        rb_p = ra_p + size_x
        cb_p = ca_p + size_y

        patch = patch[ra_p: rb_p, ca_p: cb_p]
        patch_mask = patch_mask[ra_p: rb_p, ca_p: cb_p]
        blending = blending[ra_p: rb_p, ca_p: cb_p, :]
        img[ra: rb, ca: cb, :] = img[ra: rb, ca: cb, :] * (1 - blending) + patch * blending
        if img_mask is not None:
            img_mask[ra: rb, ca: cb] = np.maximum(img_mask[ra: rb, ca: cb], patch_mask)

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
        rr = rr.tolist()
        cc = cc.tolist()
        if not isinstance(rr, Iterable) or not isinstance(cc, Iterable):
            return []
        points = [(int(r), int(c)) for r, c in zip(rr, cc)]
        points = sorted(points, key=lambda p: DrawingUtils.clockwise_angle(p, radii))
        if max_points > 0:
            max_points = min(max_points, len(points))
            idx = np.round(np.linspace(0, len(points) - 1, max_points)).astype(int)
            points = np.array(points)[idx].tolist()
        return points

    @classmethod
    def polygon(cls,
                points: Sequence[Tuple[int, int]],
                color: Union[Tuple[int, int, int], Tuple[Tuple[int, int, int]], np.ndarray]
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Draws a filled polygon on a numpy array

        :param points: vertex sequence
        :type points: Sequence[Tuple[int, int]]
        :param color: fill color, if tuple of 4 colors, a gradient is computed
        :type color: Union[Tuple[int, int, int], Tuple[Tuple[int, int, int]]]
        :return: image tensor and mask
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        min_x = min(p[0] for p in points)
        min_y = min(p[1] for p in points)
        size = (max(max(p[0] for p in points) - min_x, 1),
                max(max(p[1] for p in points) - min_y, 1))
        points = tuple((p[1] - min_y, p[0] - min_x) for p in points)
        mask = np.zeros(size)
        pil_img = Image.fromarray(mask.astype('uint8'))
        draw = ImageDraw.Draw(pil_img)
        draw.polygon(points, fill=1, outline=1)
        mask = np.array(pil_img)

        if isinstance(color, np.ndarray):
            if len(color.shape) == 2:
                color = np.stack([color] * 3, axis=-1)
            color = Resize(*mask.shape)(image=color)['image']
            poly = mask.reshape(*mask.shape, 1) * color
        elif isinstance(color[0], Sequence) and len(color) == 4:
            color = cls.gradient(mask.shape, *color)
            poly = mask.reshape(*mask.shape, 1) * color
        else:
            if isinstance(color[0], Sequence):
                color = color[0]
            poly = np.stack([mask * color[i] for i in range(3)], axis=-1).astype('uint8')

        return poly, mask

    @classmethod
    def gradient(cls,
                 size: Tuple[int, int],
                 ul: Tuple[int, int, int],
                 ur: Tuple[int, int, int],
                 ll: Tuple[int, int, int],
                 lr: Tuple[int, int, int]) -> np.ndarray:
        """Makes a rectangular color gradient from 4 vertex colors

        :param size: output rows, columns
        :type size: Tuple[int, int]
        :param ul: upper left color
        :type ul: Tuple[int, int, int]
        :param ur: upper right color
        :type ur: Tuple[int, int, int]
        :param ll: lower left color
        :type ll: Tuple[int, int, int]
        :param lr: lower right color
        :type lr: Tuple[int, int, int]
        :return: gradient image, (H, W, C)
        :rtype: np.ndarray
        """
        alpha = np.linspace(0., 1., size[1])
        U = np.outer(1. - alpha, ul) + np.outer(alpha, ur)
        L = np.outer(1. - alpha, ll) + np.outer(alpha, lr)
        alpha = np.linspace(0., 1., size[0])
        alpha = np.outer(alpha, np.ones((size[1],)))
        alpha = np.stack([alpha] * 3, axis=2)
        U = U.reshape(1, *U.shape)
        L = L.reshape(1, *L.shape)
        grad = U * (1. - alpha) + L * alpha
        return grad.astype('uint8')

    @classmethod
    def interpolate(cls, points: Sequence[Tuple[int, int]], samples: int, overlap: int = 3) -> Sequence[Tuple[int, int]]:
        """Performs a cubic interpolation of a sequence of points

        :param points: the sequence of points to interpolate
        :type points: Sequence[Tuple[int, int]]
        :param samples: the number of samples to take from the interpolated curve
        :type samples: int
        :param overlap: overlap, defaults to 3
        :type overlap: int, optional
        :return: a sequence of points obtained from cubic interpolation
        :rtype: Sequence[Tuple[int, int]]
        """
        indexes = np.arange(0, len(points) + 2 * overlap)
        interp_i = np.linspace(overlap, len(points) + overlap, samples)
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
        """Returns the angle between the vector (0, 1) and a 2d point wrt a center point c

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
