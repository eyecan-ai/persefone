import cv2
import pytest
import numpy as np

from persefone.transforms.custom.pad_if_needed_v2 import PadIfNeededV2


class TestPadIfNeededV2:

    @pytest.mark.parametrize(('params', 'img', 'expected'), (
        (
            {'min_height': 4, 'min_width': 4,
             'row_align': 'center',
             'col_align': 'center',
             'border_mode': cv2.BORDER_CONSTANT,
             },
            np.array([[1, 2],
                      [3, 4]]),
            np.array([[0, 0, 0, 0],
                      [0, 1, 2, 0],
                      [0, 3, 4, 0],
                      [0, 0, 0, 0]]),
        ),
        (
            {'min_height': 4, 'min_width': 4,
             'row_align': 'top',
             'col_align': 'left',
             'border_mode': cv2.BORDER_CONSTANT,
             },
            np.array([[1, 2],
                      [3, 4]]),
            np.array([[1, 2, 0, 0],
                      [3, 4, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]]),
        ),
        (
            {'min_height': 4, 'min_width': 4,
             'row_align': 'bottom',
             'col_align': 'right',
             'border_mode': cv2.BORDER_CONSTANT,
             },
            np.array([[1, 2],
                      [3, 4]]),
            np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 1, 2],
                      [0, 0, 3, 4]]),
        ),
    ))
    def test_pad_if_needed_v2(self, params, img, expected):
        transform = PadIfNeededV2(**params)
        padded = transform(image=img)['image']
        assert np.allclose(padded, expected)
