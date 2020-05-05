from abc import ABC, abstractmethod
import json
import pathlib
from persefone.utils.colors.color import Color

import logging

logging.basicConfig(level=logging.DEBUG)
_logger = logging.getLogger()


class Palette(ABC):
    """Generic Colors Palette

    :param ABC: Abstract Base Classe
    :type ABC: ABC
    """

    def __init__(self):
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_color(self, index) -> Color:
        raise NotImplementedError


class MaterialPalette(Palette):
    MATERIAL_COLORS = json.load(open(pathlib.Path(__file__).parent / 'stores/material.json', 'r'))
    DEFAUTL_NAMES = ['green', 'blue', 'red', 'yellow', 'purple', 'indigo', 'teal', 'lime', 'brown']
    DEFAULT_LEVEL = '500'

    def __init__(self, color_names=None, color_level=None):
        """Material Colors Palette

        :param color_names: list of color names based on MaterialColors naming convention, defaults to None
        :type color_names: list(str), optional
        :param color_level: color level (e.g. 500) based on MaterialColors naming convention, defaults to None
        :type color_level: str, optional
        """
        Palette.__init__(self)
        self.__color_names = color_names if color_names is not None else MaterialPalette.DEFAUTL_NAMES
        self.__color_level = color_level if color_level is not None else MaterialPalette.DEFAULT_LEVEL

    @property
    def size(self):
        """Get Palette size

        :return: Size of current MaterialPalette
        :rtype: int
        """
        return len(self.__color_names)

    def get_color(self, index):
        """Pick color by index within Palette

        :param index: color index
        :type index: int
        :return: picked Color or None
        :rtype: Color
        """
        if self.size == 0:
            return None
        in_index = index % len(self.__color_names)
        picked = MaterialPalette.pick_color_by_name(self.__color_names[in_index], self.__color_level)
        return Color(c=picked)

    @classmethod
    def pick_color_by_name(cls, name, level='500'):
        """Picks material color by name/evel

        :param name: color name
        :type name: str
        :param level: color level, defaults to '500'
        :type level: str, optional
        :return: hex color representation
        :rtype: str
        """
        try:
            return MaterialPalette.MATERIAL_COLORS[name][level]
        except Exception as e:
            _logger.error(f"{cls.__name__}.pick_color_by_name: {e}")
            return "#000000"
