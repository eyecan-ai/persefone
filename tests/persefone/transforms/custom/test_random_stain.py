
import numpy as np
from persefone.transforms.custom.random_stain import RandomStain
from albumentations import ToFloat


class TestRandomStain:

    def test_random_stain(self):
        random_stain = RandomStain(5, 10, 16, 128, 1, 3, 'fill', (0, 0, 0), (255, 255, 255), 20, 5, always_apply=True)
        img = np.random.rand(256, 256, 3).astype('uint8')
        stained = random_stain(image=img)['image']
        assert img.shape == stained.shape
        random_stain = RandomStain(0, 0, 16, 128, 1, 3, 'fill', (0, 0, 0), (255, 255, 255), 20, 5, always_apply=True)
        stained = random_stain(image=img)['image']
        assert np.allclose(img, stained)

    def debug(self):
        from PIL import Image
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
            stain = RandomStain(50, 100, 64, 256, 1, 3, fill_mode='solid', min_rgb=(0, 0, 0), max_rgb=(255, 255, 255))
            for cat in categories:
                img = Image.open(f'/home/luca/ae_playground_data/mvtec/{cat}/train/good/00{i}.png')
                img = np.array(img).astype('uint8')
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, axis=2)
                img = stain(image=img)['image']
                # img = self._saliency_laplace(img).astype('float32')
                img = img / img.max()
                img = Resize(1024, 1024)(image=img)['image']
                if img.dtype == 'uint8':
                    img = ToFloat()(image=img)['image']
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, axis=2)
                imgs.append(img)
        imgs = np.concatenate(imgs, axis=0)
        plt.imshow(imgs)
        plt.show()
