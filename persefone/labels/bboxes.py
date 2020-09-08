from enum import Enum, auto
from typing import List, Union
import numpy as np


class BoundingBoxType(Enum):
    """ FOrmats

    PASCAL_VOC          ->      [x_min, y_min, x_max, y_max]
    ALBUMENTATIONS      ->      [x_min, y_min, x_max, y_max](normalized)
    COCO                ->      [x_min, y_min, width, height]
    YOLO                ->      [x_center, y_center, width, height](normalized)

    """

    PASCAL_VOC = auto()
    COCO = auto()
    YOLO = auto()
    ALBUMENTATIONS = auto()


class BoundingBox(object):

    def __init__(self, data: Union[np.ndarray, list]):

        data = np.array(data).ravel()
        assert len(data) == 4

        self._data = np.array(data)

    def __eq__(self, other):
        return np.all(np.isclose(self._data, other._data))

    def as_dict(self, ref_image_size: List[int] = None) -> dict:
        """ Retrieves full dict representation for current box

        :param ref_image_size: reference image size [w,h] for normalized formats, defaults to None
        :type ref_image_size: List[int], optional
        :return: dictionay representation
        :rtype: dict
        """

        width, height = 1, 1
        if ref_image_size is not None:
            assert len(ref_image_size) == 2
            width, height = ref_image_size

        x_min, y_min, x_max, y_max = self._data
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2
        w = x_max - x_min
        h = y_max - y_min

        return {
            'x_min': x_min / width,
            'x_max': x_max / width,
            'y_min': y_min / height,
            'y_max': y_max / height,
            'x_center': x_center / width,
            'y_center': y_center / height,
            'width': w / width,
            'height': h / height,
            'image_width': width,
            'image_heiht': height
        }

    def plain_data(self, ref_image_size: List[int] = None, box_type: BoundingBoxType = BoundingBoxType.PASCAL_VOC) -> np.ndarray:
        """ Retrieves plain array data

        :param ref_image_size: reference image size [w,h], defaults to None
        :type ref_image_size: List[int], optional
        :param box_type: retrieved data format, defaults to BoundingBoxType.PASCAL_VOC
        :type box_type: BoundingBoxType, optional
        :raises NotImplementedError: data format not recognized
        :return: plain array data
        :rtype: np.ndarray
        """

        if box_type == BoundingBoxType.PASCAL_VOC:
            return self._data
        elif box_type == BoundingBoxType.COCO:
            d = self.as_dict()
            return np.array([d['x_min'], d['y_min'], d['width'], d['height']])
        elif box_type == BoundingBoxType.ALBUMENTATIONS:
            d = self.as_dict(ref_image_size=ref_image_size)
            return np.array([d['x_min'], d['y_min'], d['x_max'], d['y_max']])
        elif box_type == BoundingBoxType.YOLO:
            d = self.as_dict(ref_image_size=ref_image_size)
            return np.array([d['x_center'], d['y_center'], d['width'], d['height']])
        else:
            raise NotImplementedError(f'Type {box_type} not implemented yet!')

    @classmethod
    def build_from_type(cls, data: Union[np.ndarray, list], box_type: BoundingBoxType, image_size: List[int] = None):
        """ Builds a BoundingBox object from plain data with custom type

        :param data: plain array data
        :type data: Union[np.ndarray, list]
        :param box_type: data type
        :type box_type: BoundingBoxType
        :param image_size: reference image size [w,h], defaults to None
        :type image_size: List[int], optional
        :raises NotImplementedError: data type not recognized
        :return: BoundingBox object
        :rtype: BoundingBox
        """

        data = np.array(data).ravel()
        assert len(data) == 4

        width, height = 1000, 1000
        if image_size is not None:
            assert len(image_size) == 2
            width, height = image_size

        if box_type == BoundingBoxType.PASCAL_VOC:
            return BoundingBox(data)

        elif box_type == BoundingBoxType.COCO:
            x_min, y_min, w, h = data
            x_max = x_min + w
            y_max = y_min + h
            return BoundingBox([x_min, y_min, x_max, y_max])

        elif box_type == BoundingBoxType.ALBUMENTATIONS:
            return BoundingBox(data * [width, height, width, height])

        elif box_type == BoundingBoxType.YOLO:
            x_center, y_center, w, h = data * [width, height, width, height]
            x_min = x_center - w // 2
            y_min = y_center - h // 2
            x_max = x_min + w
            y_max = y_min + h
            return BoundingBox([x_min, y_min, x_max, y_max])
        else:
            raise NotImplementedError(f'Type {box_type} not implemented yet!')
