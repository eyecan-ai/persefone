import logging
from persefone.transforms.albumentations import AlbumentationTransformsFactory

from persefone.transforms.custom.pad_if_needed_v2 import PadIfNeededV2
from persefone.transforms.base import AbstractFactory
from persefone.transforms.custom.random_stain import RandomStain

_logger = logging.getLogger()


class CustomTransformsFactory(AbstractFactory):

    @classmethod
    def _functions_map(cls):
        return {
            'random_stain': {'f': cls._build_random_stain, 'targets': cls._targets_map()['spatial_full']},
            'pad_if_needed_v2': {'f': cls._build_pad_if_needed_v2, 'targets': cls._targets_map()['spatial_full']}
        }

    @classmethod
    def _build_random_stain(cls, **params):
        return RandomStain(
            params.get('min_holes'),
            params.get('max_holes'),
            params.get('min_size'),
            params.get('max_size'),
            params.get('min_eccentricity'),
            params.get('max_eccentricity'),
            params.get('fill_mode'),
            params.get('min_rgb'),
            params.get('max_rgb'),
            params.get('n_points'),
            params.get('perturbation_radius'),
            params.get('min_pos'),
            params.get('max_pos'),
            params.get('displacement_radius'),
            params.get('noise'),
            params.get('always_apply', False),
            params.get('p', 1)
        )

    @classmethod
    def _build_pad_if_needed_v2(cls, **params):
        return PadIfNeededV2(
            params.get('min_height', 1024),
            params.get('min_width', 1024),
            AlbumentationTransformsFactory.BORDERS.get(params.get('border_mode', 'default')),
            params.get('value'),
            params.get('mask_value'),
            params.get('row_align'),
            params.get('col_align'),
            params.get('always_apply', False),
            params.get('p', 1)
        )

    @classmethod
    def build_transform(cls, name, **params):
        functions_map = cls._functions_map()
        if name not in functions_map:
            _logger.error(f'Transform: {name} not found in {cls.__name__}')
            return None, []
        return functions_map[name]['f'](**params), functions_map[name]['targets']
