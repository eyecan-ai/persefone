import logging

from persefone.transforms.base import AbstractFactory
from persefone.transforms.custom.random_stain import RandomStain

_logger = logging.getLogger()


class CustomTransformsFactory(AbstractFactory):

    @classmethod
    def _functions_map(cls):
        return {
            'random_stain': {'f': cls._build_random_stain, 'targets': cls._targets_map()['spatial_full']},
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
            params.get('min_rgb'),
            params.get('max_rgb'),
            params.get('n_points'),
            params.get('perturbation_radius'),
            params.get('min_pos'),
            params.get('max_pos'),
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
