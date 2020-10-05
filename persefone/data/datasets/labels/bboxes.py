

from persefone.utils.colors.palettes import MaterialPalette, Palette
from typing import Any, List, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


class FieldsOptions(object):
    """ Formats

    c -> class
    s -> score
    l -> x_min [normalized]
    L -> x_min
    t -> y_min [normalized]
    T -> y_min
    r -> x_max [normalized]
    R -> x_max
    b -> y_max [normalized]
    B -> y_max
    w -> width [normalized]
    W -> width
    h -> height [normalized]
    H -> height
    u -> x_center [normalized]
    U -> x_center
    v -> y_center [normalized]
    V -> y_center
    a -> angle
    A -> angle [degrees]

    """

    FORMAT_PASCAL_VOC = 'PASCAL_VOC'
    FORMAT_YOLO = 'YOLO'
    FORMAT_YOLO_PX = 'YOLO_PX'
    FORMAT_COCO = 'COCO'
    FORMAT_ALBUMENTATIONS = 'ALBUMENTATIONS'

    FORMATS = {
        FORMAT_PASCAL_VOC: 'LTRB',  # [x_min, y_min, x_max, y_max]
        FORMAT_COCO: 'LTWH',  # [x_min, y_min, width, height]
        FORMAT_YOLO: 'uvwh',  # x_center, y_center, width, height](normalized)
        FORMAT_YOLO_PX: 'UVWH',  # x_center, y_center, width, height]
        FORMAT_ALBUMENTATIONS: 'ltrb'  # [x_min, y_min, x_max, y_max](normalized)
    }

    @classmethod
    def get_format(cls, format_name: str):
        if format_name in cls.FORMATS:
            return cls.FORMATS[format_name]
        return ''

    @classmethod
    def is_fmt_good(cls, fmt: str):
        for _, fmt_fields in cls.FORMATS.items():
            if fmt_fields in fmt:
                return True
        return False


class LabelScheleton(object):

    def __init__(self, data: np.ndarray, fmt: str):
        """ Label scheleton manages pair of (data, format) checking for
        inconsistencies and validity

        :param data: plain array data
        :type data: np.ndarray
        :param fmt: corresponding format
        :type fmt: str, optional
        """
        assert len(data) == len(fmt)
        self._data = np.array(data).ravel()
        self._fmt = fmt
        assert FieldsOptions.is_fmt_good(fmt)

    def get(self, key: str, default=None) -> Any:
        """ Gets single filed if contained in current format

        :param key: field character representation
        :type key: str
        :param default: default value in case of miss, defaults to None
        :type default: Any, optional
        :return: corresponding searched value
        :rtype: Any
        """
        try:
            idx = self._fmt.index(key)
            return self._data[idx]
        except ValueError:
            return default

    def contains_format(self, format_name: str) -> bool:
        """ Checks if this scheleton contains format

        :param format_name: target format name
        :type format_name: str
        :return: TRUE if current format contains target format
        :rtype: bool
        """
        return FieldsOptions.FORMATS[format_name] in self._fmt


class BoundingBoxLabel(object):

    def __init__(self, data: np.ndarray, fmt: str, image_size: List[int]):
        """ Generic Bounding Box Label representation

        :param data: input data
        :type data: np.ndarray
        :param fmt: input format
        :type fmt: str
        :param image_size: reference image size
        :type image_size: List[int]
        """

        self._scheleton = LabelScheleton(data, fmt)

        self._image_size = image_size
        self._label = self._scheleton.get('c', -1)
        self._score = self._scheleton.get('s', 0.0)
        self._fmt = fmt
        self._data = data

    @property
    def label(self):
        return int(self._label)

    @property
    def score(self):
        return self._score

    def in_bound(self) -> bool:
        """ Checks for coordinates inside reference image

        :return: TRUE if bbox is in bounds
        :rtype: bool
        """

        athoms = self.athoms()
        if athoms['L'] > 0 and athoms['T'] > 0:
            return athoms['R'] < self._image_size[0] - 1 and athoms['B'] < self._image_size[1] - 1
        return False

    def athoms(self) -> dict:
        """ Outputs whole fields representation as dict

        :return: dictionary with field character as key and corresponding value
        :rtype: dict
        """

        width, height = self._image_size

        x_min, y_min, x_max, y_max, x_center, y_center, w, h = [0, 0, 0, 0, 0, 0, 0, 0]

        if self._scheleton.contains_format(FieldsOptions.FORMAT_PASCAL_VOC):
            x_min = self._scheleton.get('L')
            y_min = self._scheleton.get('T')
            x_max = self._scheleton.get('R')
            y_max = self._scheleton.get('B')
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            w = x_max - x_min
            h = y_max - y_min

        elif self._scheleton.contains_format(FieldsOptions.FORMAT_COCO):
            x_min = self._scheleton.get('L')
            y_min = self._scheleton.get('T')
            w = self._scheleton.get('W')
            h = self._scheleton.get('H')
            x_max = x_min + w
            y_max = y_min + h
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2

        # [x_min, y_min, x_max, y_max](normalized)
        elif self._scheleton.contains_format(FieldsOptions.FORMAT_ALBUMENTATIONS):
            x_min = self._scheleton.get('l') * width
            y_min = self._scheleton.get('t') * height
            x_max = self._scheleton.get('r') * width
            y_max = self._scheleton.get('b') * height
            w = x_max - x_min
            h = y_max - y_min
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2

        # [x_center, y_center, width, height](normalized)
        elif self._scheleton.contains_format(FieldsOptions.FORMAT_YOLO):
            x_center = self._scheleton.get('u') * width
            y_center = self._scheleton.get('v') * height
            w = self._scheleton.get('w') * width
            h = self._scheleton.get('h') * height
            x_min = x_center - w // 2
            y_min = y_center - h // 2
            x_max = x_min + w
            y_max = y_min + h

        # [x_center, y_center, width, height](normalized)
        elif self._scheleton.contains_format(FieldsOptions.FORMAT_YOLO_PX):
            x_center = self._scheleton.get('U')
            y_center = self._scheleton.get('V')
            w = self._scheleton.get('W')
            h = self._scheleton.get('H')
            x_min = x_center - w // 2
            y_min = y_center - h // 2
            x_max = x_min + w
            y_max = y_min + h

        return {
            'l': x_min / width,
            'L': x_min,
            'r': x_max / width,
            'R': x_max,
            't': y_min / height,
            'T': y_min,
            'b': y_max / height,
            'B': y_max,
            'u': x_center / width,
            'U': x_center,
            'v': y_center / height,
            'V': y_center,
            'w': w / width,
            'W': w,
            'h': h / height,
            'H': h,
            'c': self._label,
            's': self._score
        }

    def plain_data(self, fmt: str = None) -> np.ndarray:
        """ Retrieves plain data with custom format string

        :param fmt: format string, defaults to None
        :type fmt: str, optional
        :return: plain array data
        :rtype: np.ndarray
        """

        if fmt is None:
            fmt = 'c' + FieldsOptions.get_format(FieldsOptions.FORMAT_PASCAL_VOC)

        athoms = self.athoms()
        return np.array([athoms[x] for x in fmt])


class BoundingBoxLabelDrawerParameters(object):
    """ BOunding box drawer common parameters
    """

    def __init__(self):
        self.font_size = 12

        if 'posix' in os.name:
            self.font_name = "Arial.ttf"
        else:
            self.font_name = "Pillow/Tests/fonts/FreeMono.ttf"

        self.label_size = [80, 20]
        self.default_foreground = (0, 0, 0)
        self.default_background = (255, 255, 255)


class BoundingBoxLabelDrawer(object):

    def __init__(self, palette: Palette = None, labels_map: dict = None):
        """ Creates a Bounding Box drawer

        :param palette: color palette, defaults to None
        :type palette: Palette, optional
        :param labels_map: labels map used for label2name conversion, defaults to None
        :type labels_map: dict, optional
        """

        self._palette: Palette = palette if palette is not None else MaterialPalette()
        self._labels_map = labels_map if labels_map is not None else {}

    def draw_label(self,
                   image: Union[np.ndarray, Image.Image],
                   text: str,
                   pos: List[int],
                   background: List[int] = None,
                   foreground: List[int] = None,
                   label_parameters: BoundingBoxLabelDrawerParameters = BoundingBoxLabelDrawerParameters()
                   ) -> Union[np.ndarray, Image.Image]:
        """ Draw label box on copy of input image

        :param image: input image, either a numpy array or PIL Image
        :type image: Union[np.ndarray, Image.Image]
        :param text: text to show
        :type text: str
        :param pos: lable position
        :type pos: List[int]
        :param background: label box background coolor, defaults to None
        :type background: List[int], optional
        :param foreground: label box foreground coolor, defaults to None
        :type foreground: List[int], optional
        :param label_parameters: label text parameters, defaults to BoundingBoxLabelDrawerParameters()
        :type label_parameters: BoundingBoxLabelDrawerParameters, optional
        :return: modified image
        :rtype: Union[np.ndarray, Image.Image]
        """

        image = image.copy()

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
                  bbox: BoundingBoxLabel,
                  image: Union[np.ndarray, Image.Image],
                  width: int = 2,
                  label_parameters: BoundingBoxLabelDrawerParameters = BoundingBoxLabelDrawerParameters(),
                  show_label: bool = True
                  ) -> Union[np.ndarray, Image.Image]:
        """ Draw bounding box, with labels, on copy of input image

        :param bbox: bounding box to draw
        :type bbox: BoundingBoxLabel
        :param image: input image, either a numpy array or PIL Image
        :type image: Union[np.ndarray, Image.Image]
        :param width: box line width, defaults to 2
        :type width: int, optional
        :param label_parameters: label text parameters, defaults to BoundingBoxLabelDrawerParameters()
        :type label_parameters: BoundingBoxLabelDrawerParameters, optional
        :param show_label: show or not textual informations, defaults to True
        :type show_label: bool, optional
        :return: modified image
        :rtype: Union[np.ndarray, Image.Image]
        """

        image = image.copy()

        pil = True
        if isinstance(image, np.ndarray):
            pil = False
            image = Image.fromarray(image)

        x_min, x_max, y_min, y_max, score, label = bbox.plain_data(fmt='LRTBsc')

        draw = ImageDraw.Draw(image)
        color = self._palette.get_color(bbox.label).rgb
        draw.rectangle((x_min, y_min, x_max, y_max), outline=color, width=width)

        if show_label:
            image = self.draw_label(
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
