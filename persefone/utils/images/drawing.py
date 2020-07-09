
from PIL import Image
import numpy as np
import sty


class ConsoleImage(object):

    @classmethod
    def print_image(cls, filename, scale=0.1, ratio=7 / 4, character='█'):
        """
        Args:
            filename (Str): input image
            scale (float): output scale. Default [0.1], means 48 cells width
            ratio (float): image2cells ratio. Default [7/4]
            character (char): character used to fill cells
        """
        img = Image.open(filename).convert('RGB')
        img.thumbnail([480, 480])
        S = (round(img.size[0] * scale * ratio), round(img.size[1] * scale))
        img = np.array(img.resize(S))
        for i in range(img.shape[0]):
            buf = []
            for j in range(img.shape[1]):
                sty.fg.col = sty.Style(sty.RgbFg(img[i, j, 0], img[i, j, 1], img[i, j, 2]))
                buf.append(sty.fg.col + str(character) + sty.fg.rs)
            print(''.join(buf))
