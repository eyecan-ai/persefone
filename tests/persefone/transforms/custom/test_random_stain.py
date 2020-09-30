
import numpy as np
from persefone.transforms.custom.random_stain import RandomStain


class TestRandomStain:

    def test_random_stain(self):
        random_stain = RandomStain(5, 10, 16, 128, 1, 3, 'fill', (0, 0, 0), (255, 255, 255), 20, 5, always_apply=True)
        img = np.random.rand(256, 256, 3).astype('uint8')
        stained = random_stain(image=img)['image']
        assert img.shape == stained.shape
        random_stain = RandomStain(0, 0, 16, 128, 1, 3, 'fill', (0, 0, 0), (255, 255, 255), 20, 5, always_apply=True)
        stained = random_stain(image=img)['image']
        assert np.allclose(img, stained)
