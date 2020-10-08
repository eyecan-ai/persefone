import random
from typing import Sequence, Union

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from albumentations import Lambda, ToFloat, Rotate, PadIfNeeded


def perlin(x, y):
    # permutation table
    elems = x.max() + y.max() + 1
    p = np.arange(elems, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)
    # internal coordinates
    xf = x - xi
    yf = y - yi
    # fade factors
    u = fade(xf)
    v = fade(yf)
    # noise components
    n00 = gradient(p[p[xi]+yi], xf, yf)
    n01 = gradient(p[p[xi]+yi+1], xf, yf-1)
    n11 = gradient(p[p[xi+1]+yi+1], xf-1, yf-1)
    n10 = gradient(p[p[xi+1]+yi], xf-1, yf)
    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)
    return lerp(x1, x2, v)


def lerp(a, b, x):
    "linear interpolation"
    return a + x * (b-a)


def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def gradient(h, x, y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y


def ms_perlin(size_x, size_y, scale_x, scale_y, factor_x, factor_y, iters):
    res = np.zeros((size_x, size_y))
    if isinstance(iters, int):
        iters = range(iters)
    for i in iters:
        s_x = scale_x * (factor_x ** i)
        s_y = scale_y * (factor_y ** i)
        linx = np.linspace(0, size_y / s_y, size_y, endpoint=False)
        liny = np.linspace(0, size_x / s_x, size_x, endpoint=False)
        x, y = np.meshgrid(linx, liny)
        noise = perlin(x, y)
        # factor = math.sqrt(s_x**2 + s_y**2) ** 0.1
        factor = 1
        res += noise * factor
    return res


def build_gaussian_2d(shape: Sequence[int], sigma: Union[float, Sequence[float]]) -> np.ndarray:
    """ Builds a 2D Tensor containing a centered gaussian figure
    :param shape: desired output shape
    :type shape: Sequence[int]
    :param sigma: gaussian sigma
    :type sigma: Union[float, Sequence[float, float]]
    :return: built 2D tensor
    :rtype: torch.Tensor
    """

    if not isinstance(sigma, Sequence):
        sigma = (sigma, sigma)

    m, n = [(s - 1.) / 2. for s in shape]
    # y = np.arange(-m, m + 1).unsqueeze(dim=1)
    y = np.expand_dims(np.arange(-m, m + 1), 1)
    x = np.expand_dims(np.arange(-n, n + 1), 0)
    gaussian = np.exp((- x ** 2) / (2 * sigma[1] ** 2) + (- y ** 2) / (2 * sigma[0] ** 2))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0
    return gaussian


def scratch(x, y, sigma_mult=5):
    noise = ms_perlin(x, y, 0.8, 2.5, 1.9, 2.7, 6)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = np.clip(noise, 0.5, 1.0) - 0.5
    noise = noise * build_gaussian_2d(noise.shape, [x / sigma_mult, y / sigma_mult])
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise


def random_scratch(image, **kwargs):
    min_size = 32
    max_size = 128
    min_eccentricity = 1
    max_eccentricity = 5

    size_a = random.randint(min_size, max_size)
    eccentricity = random.uniform(min_eccentricity, max_eccentricity)
    size_b = int(size_a / eccentricity)

    corr = scratch(size_b, size_a)
    corr = np.stack([corr] * 3, axis=-1)
    corr = PadIfNeeded(max(corr.shape[0], corr.shape[1]), max(corr.shape[0], corr.shape[1]),
                       border_mode=cv2.BORDER_CONSTANT)(image=corr)['image']
    corr = Rotate(360, border_mode=cv2.BORDER_CONSTANT)(image=corr)['image']

    r = random.randint(0, image.shape[0] - 1)
    c = random.randint(0, image.shape[1] - 1)

    npad = ((r, image.shape[0] - r),
            (c, image.shape[1] - c),
            (0, 0))
    corr = np.pad(corr, npad)
    image = ToFloat()(image=image)['image']
    padr = (corr.shape[0] - image.shape[0]) // 2
    padc = (corr.shape[1] - image.shape[1]) // 2
    padr_end = corr.shape[0] - image.shape[0] - padr
    padc_end = corr.shape[1] - image.shape[1] - padc
    npad = ((padr, padr_end),
            (padc, padc_end),
            (0, 0))
    image = np.pad(image, npad)
    image = image * (1 - corr) + np.ones_like(image) * corr
    image = image[padr:-padr_end, padc:-padc_end, :]
    image = (image * 255).astype('uint8')
    return image


if __name__ == '__main__':
    image = Image.open('/home/luca/test/misc/cat.jpeg')
    image = np.array(image)
    # image = np.zeros((256, 256, 3)).astype('uint8')
    transform = Lambda(image=random_scratch)
    for i in range(100):
        image = transform(image=image)['image']
    plt.imshow(image)
    plt.show()
