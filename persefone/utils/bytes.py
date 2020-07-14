
from io import BytesIO
import imageio
import numpy as np
from pathlib import Path


class DataCoding(object):

    IMAGE_CODECS = ['jpg', 'jpeg', 'png', 'tiff', 'bmp']
    NUMPY_CODECS = ['npy']
    TEXT_CODECS = ['txt']

    @classmethod
    def bytes_to_data(cls, data: bytes, data_encoding: str):
        data_encoding = data_encoding.replace('.', '')

        if data_encoding in cls.IMAGE_CODECS:
            buffer = BytesIO(data)
            return imageio.imread(buffer.getbuffer(), format=data_encoding)
        elif data_encoding in cls.NUMPY_CODECS:
            buffer = BytesIO(data)
            return np.load(buffer)
        elif data_encoding in cls.TEXT_CODECS:
            buffer = BytesIO(data)
            return np.loadtxt(buffer)
        else:
            return None

    @classmethod
    def file_to_bytes(cls, filename: str):
        filename = Path(filename)
        extension = filename.suffix.replace('.', '')
        return open(filename, 'rb').read(), extension
