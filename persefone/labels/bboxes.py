from persefone.utils.colors.color import Color
from persefone.utils.colors.palettes import MaterialPalette, Palette
from PIL import ImageDraw, Image, ImageFont
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
            'image_height': height,
            'normalized': width == 1 and height == 1
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
    def from_dict(cls, dict_data: dict) -> 'BoundingBox':
        """ Builds bounding box from dict

        :param dict_data: [description]
        :type dict_data: dict
        :return: built BoundingBox
        :rtype: BoundingBox
        """

        return BoundingBox([
            dict_data['x_min'] * dict_data['image_width'],
            dict_data['y_min'] * dict_data['image_height'],
            dict_data['x_max'] * dict_data['image_width'],
            dict_data['y_max'] * dict_data['image_height']
        ])

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


class BoundingBoxWithLabelAndScore(BoundingBox):

    def __init__(self, data: Union[np.ndarray, list]):

        data = np.array(data).ravel()
        assert len(data) == 6

        super(BoundingBoxWithLabelAndScore, self).__init__(data=data[:4])
        self._label = data[5]
        self._score = data[4]

    @property
    def label(self):
        return int(self._label)

    @property
    def score(self):
        return self._score

    def as_dict(self, ref_image_size: List[int] = None) -> dict:
        """ Retrieves full dict representation for current box

        :param ref_image_size: reference image size [w,h] for normalized formats, defaults to None
        :type ref_image_size: List[int], optional
        :return: dictionay representation
        :rtype: dict
        """

        d = super().as_dict(ref_image_size)
        d.update({
            'label': self.label,
            'score': self.score
        })
        return d

    @classmethod
    def from_dict(cls, dict_data: dict) -> 'BoundingBox':
        """ Builds bounding box from dict

        :param dict_data: [description]
        :type dict_data: dict
        :return: built BoundingBox
        :rtype: BoundingBox
        """

        return BoundingBoxWithLabelAndScore([
            dict_data['x_min'] * dict_data['image_width'],
            dict_data['y_min'] * dict_data['image_height'],
            dict_data['x_max'] * dict_data['image_width'],
            dict_data['y_max'] * dict_data['image_height'],
            dict_data['score'],
            dict_data['label']
        ])

    def __eq__(self, other):
        return np.all(np.isclose(self._data[: 5], other._data[: 5])) and int(self.label) == int(other.label)

    def plain_data(self, ref_image_size: List[int] = None, box_type: BoundingBoxType = BoundingBoxType.PASCAL_VOC, with_score: bool = True) -> np.ndarray:
        """ Retrieves plain array data

        :param ref_image_size: reference image size [w,h], defaults to None
        :type ref_image_size: List[int], optional
        :param box_type: retrieved data format, defaults to BoundingBoxType.PASCAL_VOC
        :type box_type: BoundingBoxType, optional
        :param with_score: return score value
        :type with_score: bool, optional
        :raises NotImplementedError: data format not recognized
        :return: plain array data
        :rtype: np.ndarray
        """

        if with_score:
            return np.array(super().plain_data(ref_image_size, box_type).tolist() + [self.score, self.label])
        else:
            return np.array(super().plain_data(ref_image_size, box_type).tolist() + [self.label])

    def export_label(self, fmt='bl', ref_image_size: List[int] = None, box_type: BoundingBoxType = BoundingBoxType.PASCAL_VOC) -> List[int]:
        """ Exports a plain label format

        :param fmt: format chars [b|l|s] b=box, l=label, s=score, defaults to 'bl'
        :type fmt: str, optional
        :param ref_image_size: reference image size, defaults to None
        :type ref_image_size: List[int], optional
        :param box_type: box data type, defaults to BoundingBoxType.PASCAL_VOC
        :type box_type: BoundingBoxType, optional
        :return: plain list representingn full label
        :rtype: List[int]
        """

        plain_data = self.plain_data(ref_image_size=ref_image_size, box_type=box_type, with_score=True)
        data_map = {
            'b': plain_data[:4].tolist(),
            'l': [int(plain_data[5])],
            's': [plain_data[4]]
        }
        exported_label = []
        for c in fmt:
            exported_label += data_map[c]
        return exported_label

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
        assert len(data) == 6

        score, label = data[4: 6]
        bbox: BoundingBox = super(BoundingBoxWithLabelAndScore, cls).build_from_type(data[: 4], box_type, image_size)

        plain_data = bbox.plain_data().tolist() + [score, label]
        return BoundingBoxWithLabelAndScore(plain_data)


class BoundingBoxDrawerLabelParameters(object):

    def __init__(self):
        self.font_size = 12
        self.font_name = "Pillow/Tests/fonts/FreeMono.ttf"
        self.label_size = [80, 20]
        self.default_foreground = (0, 0, 0)
        self.default_background = (255, 255, 255)


class BoundingBoxDrawer(object):

    def __init__(self, palette: Palette = None, labels_map: dict = None):

        self._palette: Palette = palette if palette is not None else MaterialPalette()
        self._labels_map = labels_map if labels_map is not None else {}

    def draw_label(self,
                   image: Union[np.ndarray, Image.Image],
                   text: str,
                   pos: List[int],
                   background: List[int] = None,
                   foreground: List[int] = None,
                   label_parameters: BoundingBoxDrawerLabelParameters = BoundingBoxDrawerLabelParameters()
                   ):

        pil = True
        if isinstance(image, np.ndarray):
            pil = False
            image = Image.fromarray(image)

        W, H = label_parameters.label_size

        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(label_parameters.font_name, label_parameters.font_size)
        w, h = draw.textsize(text, font=font)

        draw.rectangle([pos[0], pos[1], pos[0] + W, pos[1] + H], fill=background)
        draw.text((pos[0] + (W - w) // 2, pos[1] + (H - h) // 2), text, fill=foreground, font=font)

        if not pil:
            image = np.array(image)
        return image

    def draw_bbox(self,
                  bbox: BoundingBoxWithLabelAndScore,
                  image: np.ndarray,
                  width: int = 2,
                  label_parameters: BoundingBoxDrawerLabelParameters = BoundingBoxDrawerLabelParameters(),
                  show_label: bool = True
                  ):

        pil = True
        if isinstance(image, np.ndarray):
            pil = False
            image = Image.fromarray(image)

        x_min, x_max, y_min, y_max, score, label = bbox.plain_data()

        draw = ImageDraw.Draw(image)
        color = self._palette.get_color(bbox.label).rgb
        draw.rectangle(bbox.plain_data()[:4].tolist(), outline=color, width=width)

        if show_label:
            self.draw_label(
                image,
                f"{int(label)}[{score:.2f}]",
                [x_min, y_max, x_max, y_max + 30],
                background=color,
                foreground=None,
                label_parameters=label_parameters
            )

        if not pil:
            image = np.array(image)
        return image
