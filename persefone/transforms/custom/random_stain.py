import copy
import math
import random
from typing import Tuple

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
    :param perturbation_radius: points displacement radius
    :type perturbation_radius: int
    :param min_pos: minimum rc position, default None
    :type min_pos: Tuple[int, int]
    :param max_pos: maximum rc position, default None
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

    def apply(self, image, **params):
        out = copy.deepcopy(image)
        n_holes = random.randint(self.min_holes, self.max_holes)
        for i in range(n_holes):
            size_a = random.randint(self.min_size, self.max_size)
            eccentricity = random.uniform(self.min_eccentricity, self.max_eccentricity)
            size_b = int(size_a / eccentricity)
            angle = random.random() * math.pi * 2
            points = DrawingUtils.ellipse((size_a, size_b), angle, self.n_points)
            displacement = self.perturbation_radius
            if displacement < 0:
                displacement = int(math.sqrt((size_a ** 2 + size_b ** 2) / 2) * math.pi / len(points))
            points = DrawingUtils.random_displace(points, displacement)
            points = DrawingUtils.interpolate(points, len(points) * 10)
            if self.min_rgb is not None and self.max_rgb is not None:
                color = tuple(random.uniform(self.min_rgb[i], self.max_rgb[i]) for i in range(3))
            else:
                pick = (
                    random.randint(self.min_pos[0] if self.min_pos is not None else 0,
                                   self.max_pos[0] if self.max_pos is not None else out.shape[0] - 1),
                    random.randint(self.min_pos[1] if self.min_pos is not None else 0,
                                   self.max_pos[1] if self.max_pos is not None else out.shape[1] - 1)
                )
                color = tuple(image[pick[0], pick[1]])
            corr, corr_mask = DrawingUtils.polygon(points, color)
            pos = (
                random.randint(self.min_pos[0] if self.min_pos is not None else 0,
                               self.max_pos[0] if self.max_pos is not None else out.shape[0] - 1),
                random.randint(self.min_pos[1] if self.min_pos is not None else 0,
                               self.max_pos[1] if self.max_pos is not None else out.shape[1] - 1)
            )
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
            'max_pos'
        )


if __name__ == '__main__':
    import torch
    import numpy as np
    from ae_playground.utils.tensor_utils import TensorUtils
    import matplotlib.pyplot as plt

    imgs = []
    stain = RandomStain(
        3, 8, 16, 64, 1, 2,
        min_pos=(0, 20),
        max_pos=(20, 40),
        always_apply=True
    )
    for i in range(16):
        img = stain(image=np.random.random((256, 256, 3)) + np.random.randint(0, 256, (3,)))['image']
        imgs.append(np.transpose(img, (2, 0, 1)))
    imgs = np.stack(imgs, axis=0)
    imgs = torch.from_numpy(imgs)
    plt.imshow(TensorUtils.to_numpy(TensorUtils.make_images(imgs)))
    plt.show()
