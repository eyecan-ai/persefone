import copy
import math
from math import sqrt
import random
from typing import Tuple

import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

from persefone.transforms.custom.drawing_utils import DrawingUtils


class RandomStain(ImageOnlyTransform):
    """Adds random stain patches to a target image

    :param min_holes: minimum number of stains
    :type min_holes: int
    :param max_holes: maximum number of stains
    :type max_holes: int
    :param min_size: minimum ellipse major axis size
    :type min_size: int
    :param max_size: maximum ellipse major axis size
    :type max_size: int
    :param min_eccentricity: minimum eccentricity
    :type min_eccentricity: float
    :param max_eccentricity: maximum eccentricity
    :type max_eccentricity: float
    :param min_rgb: minimum rgb value, default None
    :type min_rgb: Tuple[int, int, int]
    :param max_rgb: maximum rgb value, default None
    :type max_rgb: Tuple[int, int, int]
    :param n_points: number of points per stain, default 20
    :type n_points: int
    :param perturbation_radius: points displacement radius, default -1
    :type perturbation_radius: int
    :param min_pos: minimum rc position, default None
    :type min_pos: Tuple[int, int]
    :param max_pos: maximum rc position, default None
    :type max_pos: Tuple[int, int]
    :param displacement_radius: maximum stain displacement radius,
    if negative, the radius is computed as sqrt(image_size) / -displacement_radius, default -10
    :type max_pos: Tuple[int, int]
    """

    def __init__(self,
                 min_holes: int,
                 max_holes: int,
                 min_size: int,
                 max_size: int,
                 min_eccentricity: float,
                 max_eccentricity: float,
                 min_rgb: Tuple[int, int, int] = None,
                 max_rgb: Tuple[int, int, int] = None,
                 n_points: int = 20,
                 perturbation_radius: int = -1,
                 min_pos: Tuple[int, int] = None,
                 max_pos: Tuple[int, int] = None,
                 displacement_radius: int = -10,
                 always_apply=False, p=1.0) -> None:
        super(RandomStain, self).__init__(always_apply, p)
        self.min_holes = min_holes
        self.max_holes = max_holes
        self.min_size = min_size
        self.max_size = max_size
        self.min_eccentricity = min_eccentricity
        self.max_eccentricity = max_eccentricity
        self.min_rgb = min_rgb
        self.max_rgb = max_rgb
        self.n_points = n_points
        self.perturbation_radius = perturbation_radius
        self.min_pos = min_pos
        self.max_pos = max_pos
        self.displacement_radius = displacement_radius

    def apply(self, image, **params):

        def _saliency(image: np.ndarray) -> np.ndarray:
            gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) * 255).astype(np.uint8)
            edges = cv2.Canny(gray, 10, 120)
            kernel = np.ones((5, 5), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1) / 255.
            saliency = edges
            # saliency = cv2.GaussianBlur(edges, (51, 51), 0)
            if saliency.sum() > 0.:
                saliency = saliency / saliency.sum()
            else:
                saliency[::] = 1. / saliency.size
            return saliency

        def _rand_pos(p: np.ndarray = None, disp: int = 0) -> Tuple[int, int]:
            min_r = self.min_pos[0] if self.min_pos is not None else 0
            max_r = self.max_pos[0] if self.max_pos is not None else out.shape[0] - 1
            min_c = self.min_pos[1] if self.min_pos is not None else 0
            max_c = self.max_pos[1] if self.max_pos is not None else out.shape[1] - 1
            if p is None or p.sum() <= 0.:
                r = random.randint(min_r, max_r)
                c = random.randint(min_c, max_c)
            else:
                c = np.random.choice(np.arange(p.shape[1]), p=p.sum(0))
                r = np.random.choice(np.arange(p.shape[0]), p=p[:, c] / p[:, c].sum())
            r += random.randint(-disp // 2, disp // 2)
            c += random.randint(-disp // 2, disp // 2)
            r = min(max_r, r)
            c = min(max_c, c)
            r = max(min_r, r)
            c = max(min_c, c)
            return r, c

        out = copy.deepcopy(image)
        n_holes = random.randint(self.min_holes, self.max_holes)
        saliency = _saliency(image)
        # import matplotlib.pyplot as plt
        # plt.imshow(saliency)
        # plt.show()
        for i in range(n_holes):
            size_a = random.randint(self.min_size, self.max_size)
            eccentricity = random.uniform(self.min_eccentricity, self.max_eccentricity)
            size_b = int(size_a / eccentricity)
            angle = random.random() * math.pi * 2

            # Compute perturbation
            pert = self.perturbation_radius
            if pert < 0:
                pert = int(math.sqrt((size_a ** 2 + size_b ** 2) / 2) * math.pi / self.n_points)

            # Compute Points
            points = DrawingUtils.ellipse((size_a, size_b), angle, self.n_points)
            points = DrawingUtils.random_displace(points, pert)
            points = DrawingUtils.interpolate(points, len(points) * 10)

            # Compute displacement
            if self.displacement_radius >= 0:
                disp = self.displacement_radius
            else:
                size = image.shape[0] * image.shape[1]
                disp = sqrt(size) / -self.displacement_radius

            # Compute RGB
            if self.min_rgb is not None and self.max_rgb is not None:
                color = tuple(random.uniform(self.min_rgb[i], self.max_rgb[i]) for i in range(3))
            else:
                pick = _rand_pos(saliency, disp)
                color = image[pick[0], pick[1]]
                color = tuple(color.reshape(color.size))

            # Create patch
            corr, corr_mask = DrawingUtils.polygon(points, color)
            pos = _rand_pos(saliency, disp)
            pos = tuple(pos[i] - corr.shape[i] // 2 for i in range(2))

            # Apply patch
            DrawingUtils.apply_patch(out, None, corr, corr_mask, pos)
        return out

    def get_transform_init_args_names(self):
        return (
            'min_holes',
            'max_holes',
            'min_size',
            'max_size',
            'min_eccentricity',
            'max_eccentricity',
            'min_rgb',
            'max_rgb',
            'n_points',
            'perturbation_radius',
            'min_pos',
            'max_pos',
            'displacement_radius'
        )

    def debug(self):
        import torch
        from PIL import Image
        from ae_playground.utils.tensor_utils import TensorUtils
        import matplotlib.pyplot as plt

        imgs = []
        for i in range(16):
            img = Image.open('/home/luca/ae_playground_data/mvtec/wood/train/good/006.png')
            img = np.array(img).astype('uint8')
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=2)
            img = self(image=img)['image']
            imgs.append(np.transpose(img, (2, 0, 1)))
        imgs = np.stack(imgs, axis=0)
        imgs = torch.from_numpy(imgs.astype('float32') / 255.)
        plt.imshow(TensorUtils.to_numpy(TensorUtils.make_images(imgs)))
        plt.show()
