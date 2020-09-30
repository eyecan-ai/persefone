import copy
import math
from math import sqrt
import random
from typing import Tuple

import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations import GaussNoise, ToFloat

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

    # old_params: low_t = 10, high_t = 120, dilate = 5
    def _saliency(self,
                  image: np.ndarray,
                  low_t: float = 15.,
                  high_t: float = 50.,
                  dilate: int = 5) -> np.ndarray:
        gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) * 255).astype(np.uint8)
        edges = cv2.Canny(gray, low_t, high_t)
        kernel = np.ones((dilate, dilate), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1) / 255.
        saliency = edges
        return saliency

    def _saliency_laplace(self,
                          img: np.ndarray,
                          smooth_kernel: int = 15,
                          laplacian_kernel: int = 5,
                          erode_kernel: int = 5):
        img = img.copy()
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, ksize=smooth_kernel)
        saliency = cv2.Laplacian(gray, cv2.CV_16S, ksize=laplacian_kernel)
        saliency[saliency < 0.5] = 0
        saliency = (saliency * 255.).astype(np.uint8)
        saliency = cv2.erode(saliency, kernel=np.ones((erode_kernel, erode_kernel), np.uint8))
        return saliency / 255.

    def _rand_pos(self,
                  min_r: int, max_r: int, min_c: int, max_c: int,
                  p: np.ndarray = None,
                  disp: int = 0) -> Tuple[int, int]:
        if p is None or p.sum() <= 0.:
            r = random.randint(min_r, max_r)
            c = random.randint(min_c, max_c)
        else:
            if p.sum() > 0.:
                p = p / p.sum()
            else:
                p[::] = 1. / p.size
            c = np.random.choice(np.arange(p.shape[1]), p=p.sum(0))
            r = np.random.choice(np.arange(p.shape[0]), p=p[:, c] / p[:, c].sum())
        r += random.randint(-disp // 2, disp // 2)
        c += random.randint(-disp // 2, disp // 2)
        r = min(max_r, r)
        c = min(max_c, c)
        r = max(min_r, r)
        c = max(min_c, c)
        return r, c

    def apply(self, image, **params):
        out = copy.deepcopy(image)
        n_holes = random.randint(self.min_holes, self.max_holes)
        saliency = self._saliency_laplace(image)
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

            # Position bounds
            min_r = self.min_pos[0] if self.min_pos is not None else 0
            max_r = self.max_pos[0] if self.max_pos is not None else out.shape[0] - 1
            min_c = self.min_pos[1] if self.min_pos is not None else 0
            max_c = self.max_pos[1] if self.max_pos is not None else out.shape[1] - 1

            # Compute RGB
            if self.min_rgb is not None and self.max_rgb is not None:
                colors = tuple(random.uniform(self.min_rgb[i], self.max_rgb[i]) for i in range(3))
            else:
                # center_pos = self._rand_pos(min_r, max_r, min_c, max_c, saliency, disp)
                # p = np.zeros_like(saliency)
                # r = int(disp) // 2
                # p[center_pos[0] - r: center_pos[0] + r, center_pos[1] - r: center_pos[1] + r] = 1.
                pos_list = [self._rand_pos(min_r, max_r, min_c, max_c, saliency, disp) for _ in range(4)]
                colors = tuple(image[pos[0], pos[1]] for pos in pos_list)
                colors = tuple(tuple(color.reshape(color.size)) for color in colors)

            # Create patch
            corr, corr_mask = DrawingUtils.polygon(points, colors)
            pos = self._rand_pos(min_r, max_r, min_c, max_c, saliency, disp)
            pos = tuple(pos[i] - corr.shape[i] // 2 for i in range(2))

            # Add noise to gradient
            corr = GaussNoise(var_limit=(4, 6), p=1.)(image=corr)['image']

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
        from albumentations import Resize
        import matplotlib.pyplot as plt

        imgs = []
        categories = [
            'bottle',
            'cable',
            'capsule',
            # 'carpet',
            'grid',
            'hazelnut',
            # 'leather',
            'metal_nut',
            'pill',
            'screw',
            # 'tile',
            'toothbrush',
            'transistor',
            # 'wood',
            'zipper'
        ]
        for i in range(0, 1):
            for cat in categories:
                img = Image.open(f'/home/luca/ae_playground_data/mvtec/{cat}/train/good/00{i}.png')
                img = np.array(img).astype('uint8')
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, axis=2)
                img = self(image=img)['image']
                # img = self._saliency_laplace(img).astype('float32')
                img = img / img.max()
                img = Resize(1024, 1024)(image=img)['image']
                if img.dtype == 'uint8':
                    img = ToFloat()(image=img)['image']
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, axis=2)
                img = np.transpose(img, (2, 0, 1))
                imgs.append(img)
        imgs = np.stack(imgs, axis=0)
        imgs = torch.from_numpy(imgs)
        plt.imshow(TensorUtils.to_numpy(TensorUtils.make_images(imgs)))
        plt.show()
