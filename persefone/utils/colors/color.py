import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
_logger = logging.getLogger()


class Color(object):

    def __init__(self, c):
        """Object for Color manipulation

        :param c: input color. Can be an hex string like '#ff0022' or int tuple like (120,22,33)
        :type c: str or tuple
        """
        self.__hex = "000000"
        if isinstance(c, str):
            if c.startswith('#'):
                self.hex = c
        if isinstance(c, tuple):
            if len(c) == 3 or len(c) == 4:
                self.hex = Color.rgb2hex(c)

    @classmethod
    def hex2rgb(cls, h):
        """Converts HEX color string to RGB tuple

        :param h: hex color string
        :type h: str
        :return: rgb tuple
        :rtype: tuple
        """
        try:
            h = h.lstrip('#')
            if len(h) == 6:
                return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
            if len(h) == 8:
                return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4, 6))
            return (0, 0, 0)
        except Exception as e:
            _logger.error(f"{cls.__name__}.hex2rgb: {e}")
            return (0, 0, 0)

    @classmethod
    def rgb2hex(cls, rgb):
        """Converts RGB tuple color to hex representatoin

        :param rgb: rgb tuple
        :type rgb: tuple
        :return: hex string representation
        :rtype: str
        """
        try:
            hx = ''.join(["%02x" % np.clip(int(c), 0, 255) for c in rgb])
            return f'#{hx}'
        except Exception as e:
            _logger.error(f"{cls.__name__}.rgb2hex: {e}")
            return '#000000'

    @property
    def hex(self):
        """Gets current hex representation

        :return: hex representation with left '#'
        :rtype: str
        """
        return '#' + self.__hex

    @hex.setter
    def hex(self, h):
        """Hex property setter, manager wrong format

        :param h: hex string representation
        :type h: str
        """
        h = h.lstrip('#')
        self.__hex = h
        if Color.rgb2hex(Color.hex2rgb(self.hex)) != self.hex:
            self.hex = "#000000"

    @property
    def rgb(self):
        """Gets current RGB tuple representation in [0,255]

        :return: RGB tuple representation
        :rtype: tuple
        """
        return Color.hex2rgb(self.__hex)

    @property
    def rgbf(self):
        """Gets current RGB tuple float representation  in [0.,1.]

        :return: RGB tuple float representation
        :rtype: tuple
        """
        return tuple(self.rgb_array.astype(float) / 255.)

    @property
    def rgbf_array(self):
        """Gets current RGB array float representation in [0.,1.0]

        :return: RGB array float representation
        :rtype: np.array
        """
        return np.array(self.rgbf)

    @property
    def rgb_array(self):
        """Gets current RGB array int representation in [0,255]

        :return: RGB tuple int representation
        :rtype: tuple
        """
        return np.array(self.rgb)

    def has_alpha(self):
        """Checks if Color has Alpha channel

        :return: TRUE if alpha channel is present
        :rtype: bool
        """
        return len(self.__hex) == 8

    def add_alpha(self, a=1.0):
        """Adds alpha channel to current color

        :param a: initial alpha value in [0.,1.0], defaults to 1.0
        :type a: float, optional
        """
        a = np.clip(a, 0.0, 1.0)
        if not self.has_alpha():
            self.hex = self.__hex + 'ff'

        rgb_array = self.rgb_array
        rgb_array[3] = np.clip(int(a * 255), 0, 255)
        self.hex = Color.rgb2hex(rgb_array)
