import cv2
from .base import AbstractFactory
from persefone.utils.pyutils import get_arg
import albumentations as A
import logging

_logger = logging.getLogger()


class AlbumentationTransformsFactory(AbstractFactory):

    REGISTERED_TRANSFORMS = {
        'resize': {},
        'rotate': {},
        'padding': {}
    }

    INTERPOLATIONS = {
        'default': cv2.INTER_NEAREST,
        'none': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC
    }

    BORDERS = {  # https://docs.opencv.org/3.4/d2/de8/group__core__array.html
        'default': cv2.BORDER_CONSTANT,
        'constant': cv2.BORDER_CONSTANT,
        'reflect': cv2.BORDER_REFLECT,
        'reflect101': cv2.BORDER_REFLECT101,
        'replicate': cv2.BORDER_REPLICATE,
        'wrap': cv2.BORDER_WRAP
    }

    @classmethod
    def _get_interpolation_value(cls, name):
        if name in cls.INTERPOLATIONS:
            return cls.INTERPOLATIONS[name]
        return cls.INTERPOLATIONS['default']

    @classmethod
    def _get_borders_value(cls, name):
        if name in cls.BORDERS:
            return cls.BORDERS[name]
        return cls.BORDERS['default']

    @classmethod
    def _functions_map(cls):
        return {
            'resize': {'f': cls._build_resize_transform, 'targets': cls._targets_map()['spatial_full']},
            'resize_longest': {'f': cls._build_resize_longest_transform, 'targets': cls._targets_map()['spatial_full']},
            'resize_smallest': {'f': cls._build_resize_smallest_transform, 'targets': cls._targets_map()['spatial_full']},
            'rotate': {'f': cls._build_rotate_transform, 'targets': cls._targets_map()['spatial_full']},
            'shift_scale_rotate': {'f': cls._build_random_shift_scale_rotate, 'targets': cls._targets_map()['spatial_full']},
            'crop': {'f': cls._build_crop_transform, 'targets': cls._targets_map()['spatial_full']},
            'random_crop': {'f': cls._build_random_crop_transform, 'targets': cls._targets_map()['spatial_full']},
            'random_brightness_contrast': {'f': cls._build_random_brightness_contrast, 'targets': cls._targets_map()['pixels']},
            'random_hsv': {'f': cls._build_random_hsv, 'targets': cls._targets_map()['pixels']},
            'random_grid_shuffle': {'f': cls._build_random_grid_shuffle, 'targets': cls._targets_map()['spatial_half']},
            'horizontal_flip': {'f': cls._build_horizontal_flip, 'targets': cls._targets_map()['spatial_full']},
            'vertical_flip': {'f': cls._build_vertical_flip, 'targets': cls._targets_map()['spatial_full']},
            'flip': {'f': cls._build_flip, 'targets': cls._targets_map()['spatial_full']},
            'normalize': {'f': cls._build_normalize, 'targets': cls._targets_map()['spatial_full']},
            'coarse_dropout': {'f': cls._build_coarse_dropout, 'targets': cls._targets_map()['spatial_full']},
            'pad_if_needed': {'f': cls._build_pad_if_needed, 'targets': cls._targets_map()['spatial_full']},
        }

    @classmethod
    def _build_resize_transform(cls, **params):
        size = get_arg(params, 'size', [256, 256])
        p = get_arg(params, 'p', 1.0)
        always = get_arg(params, 'always_apply', True)
        interpolation = cls._get_interpolation_value(get_arg(params, 'interpolation', 'none'))
        return A.Resize(
            height=size[1],
            width=size[0],
            interpolation=interpolation,
            always_apply=always,
            p=p
        )

    @classmethod
    def _build_resize_longest_transform(cls, **params):
        return A.LongestMaxSize(
            max_size=params.get('max_size', 1024),
            interpolation=cls._get_interpolation_value(get_arg(params, 'interpolation', 'none')),
            always_apply=get_arg(params, 'always_apply', True),
            p=get_arg(params, 'p', 1.0)
        )

    @classmethod
    def _build_resize_smallest_transform(cls, **params):
        return A.SmallestMaxSize(
            max_size=params.get('max_size', 1024),
            interpolation=cls._get_interpolation_value(get_arg(params, 'interpolation', 'none')),
            always_apply=get_arg(params, 'always_apply', True),
            p=get_arg(params, 'p', 1.0)
        )

    @classmethod
    def _build_rotate_transform(cls, **params):
        return A.Rotate(
            limit=get_arg(params, 'limit', [-45, 45]),
            interpolation=cls._get_interpolation_value(get_arg(params, 'interpolation', 'none')),
            border_mode=cls._get_borders_value(get_arg(params, 'border_mode', 'constant')),
            value=get_arg(params, 'value', 0),
            mask_value=get_arg(params, 'mask_value', 0),
            always_apply=get_arg(params, 'always_apply', True),
            p=get_arg(params, 'p', 1.0)
        )

    @classmethod
    def _build_crop_transform(cls, **params):
        box = get_arg(params, 'box', [0, 0, 256, 256])
        return A.Crop(
            x_min=box[0],
            y_min=box[1],
            x_max=box[2],
            y_max=box[3],
            always_apply=get_arg(params, 'always_apply', True),
            p=get_arg(params, 'p', 1.0)
        )

    @classmethod
    def _build_random_crop_transform(cls, **params):
        size = get_arg(params, 'size', [64, 64])
        return A.RandomCrop(
            height=size[1],
            width=size[0],
            always_apply=get_arg(params, 'always_apply', True),
            p=get_arg(params, 'p', 1.0)
        )

    @classmethod
    def _build_random_brightness_contrast(cls, **params):
        return A.RandomBrightnessContrast(
            brightness_limit=get_arg(params, 'brightness', 0.0),
            contrast_limit=get_arg(params, 'contrast', 0.0),
            brightness_by_max=get_arg(params, 'brighntess_by_max', True),
            always_apply=get_arg(params, 'always_apply', True),
            p=get_arg(params, 'p', 1.0)
        )

    @classmethod
    def _build_random_hsv(cls, **params):  # TODO: hue/sat/val values in albumentations are in [0,255] range, bad!
        return A.HueSaturationValue(
            hue_shift_limit=get_arg(params, 'hue_shift_limit', 0.0),
            sat_shift_limit=get_arg(params, 'sat_shift_limit', 0.0),
            val_shift_limit=get_arg(params, 'val_shift_limit', 0.0),
            always_apply=get_arg(params, 'always_apply', True),
            p=get_arg(params, 'p', 1.0)
        )

    @classmethod
    def _build_random_shift_scale_rotate(cls, **params):
        return A.ShiftScaleRotate(
            shift_limit=get_arg(params, 'shift_limit', 0.0),
            scale_limit=get_arg(params, 'scale_limit', 0.0),
            rotate_limit=get_arg(params, 'rotate_limit', 0),
            interpolation=cls._get_interpolation_value(get_arg(params, 'interpolation', 'linear')),
            border_mode=cls._get_borders_value(get_arg(params, 'border_mode', 'replicate')),
            value=get_arg(params, 'value', 0),
            mask_value=get_arg(params, 'mask_value', 0),
            always_apply=get_arg(params, 'always_apply', True),
            p=get_arg(params, 'p', 1.0)
        )

    @classmethod
    def _build_random_grid_shuffle(cls, **params):
        return A.RandomGridShuffle(
            grid=get_arg(params, 'grid', [3, 3]),
            always_apply=get_arg(params, 'always_apply', True),
            p=get_arg(params, 'p', 1.0)
        )

    @classmethod
    def _build_horizontal_flip(cls, **params):
        return A.HorizontalFlip(
            always_apply=get_arg(params, 'always_apply', True),
            p=get_arg(params, 'p', 1.0)
        )

    @classmethod
    def _build_vertical_flip(cls, **params):
        return A.VerticalFlip(
            always_apply=get_arg(params, 'always_apply', True),
            p=get_arg(params, 'p', 1.0)
        )

    @classmethod
    def _build_flip(cls, **params):
        return A.Flip(
            always_apply=get_arg(params, 'always_apply', True),
            p=get_arg(params, 'p', 1.0)
        )

    @classmethod
    def _build_normalize(cls, **params):
        return A.Normalize(
            mean=get_arg(params, 'mean', (0.485, 0.456, 0.406)),
            std=get_arg(params, 'std', (0.229, 0.224, 0.225)),
            max_pixel_value=get_arg(params, 'max_pixel_value', 255.0),
            always_apply=get_arg(params, 'always_apply', False),
            p=get_arg(params, 'p', 1.0)
        )

    @classmethod
    def _build_coarse_dropout(cls, **params):
        return A.CoarseDropout(
            max_holes=get_arg(params, 'max_holes', 8),
            max_width=get_arg(params, 'max_width', 8),
            max_height=get_arg(params, 'max_height', 8),
            min_holes=get_arg(params, 'min_holes', None),
            min_width=get_arg(params, 'min_width', None),
            min_height=get_arg(params, 'min_height', None),
            fill_value=get_arg(params, 'fill_value', 0),
            always_apply=get_arg(params, 'always_apply', False),
            p=get_arg(params, 'p', 0.5)
        )

    @classmethod
    def _build_pad_if_needed(cls, **params):
        return A.PadIfNeeded(
            min_height=get_arg(params, 'min_height', 1000),
            min_width=get_arg(params, 'min_width', 1000),
            border_mode=cls._get_borders_value(get_arg(params, 'border_mode', 'constant')),
            value=get_arg(params, 'value', 0),
            mask_value=get_arg(params, 'value', 0),
            always_apply=get_arg(params, 'always_apply', False),
            p=get_arg(params, 'p', 0.5)
        )

    @classmethod
    def build_transform(cls, name, **params):
        functions_map = cls._functions_map()
        if name not in functions_map:
            _logger.error(f'Transform: {name} not found in {cls.__name__}')
            return None, []
        return functions_map[name]['f'](**params), functions_map[name]['targets']
