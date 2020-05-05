import pytest
from persefone.transforms.factory import TransformsFactory, AlbumentationTransformsFactory


@pytest.mark.transforms
class TestTransformsFactory(object):

    @pytest.fixture
    def augmentations_folder(self):
        import pathlib
        return pathlib.Path(__file__).parent / '../../sample_data/augmentations'

    def test_load_full_configuration_sample(self, augmentations_folder):

        cfg_file = augmentations_folder / 'full_augmentations.yml'
        composition = TransformsFactory.parse_file(cfg_file)
        composition_dict = composition.get_dict_with_id()

        # Check for composition consistency
        assert 'transforms' in composition_dict, "composition dict is wrong!"
        transforms = composition_dict['transforms']

        # Creates Expected Transforms List. Has to match with YAML Configuration file list
        name_field = '__class_fullname__'
        expected_transforms = [
            {
                'name': 'albumentations.augmentations.transforms.Resize',
                'params': {'height': 516, 'width': 481, 'always_apply': True}
            },
            {
                'name': 'albumentations.augmentations.transforms.Rotate',
                'params': {
                    'limit': (-23, 42),
                    'p': 0.2,
                    'always_apply': True,
                    'interpolation': AlbumentationTransformsFactory.INTERPOLATIONS['linear'],
                    'border_mode': AlbumentationTransformsFactory.BORDERS['replicate'],
                    'value': [255, 0, 200],
                    'mask_value': 123
                }
            },
            {
                'name': 'albumentations.augmentations.transforms.Crop',
                'params': {'x_min': 10, 'y_min': 20, 'x_max': 30, 'y_max': 40, 'p': 0.3, 'always_apply': True}
            },
            {
                'name': 'albumentations.augmentations.transforms.RandomCrop',
                'params': {'height': 33, 'width': 44, 'p': 0.4, 'always_apply': True}
            },
            {
                'name': 'albumentations.augmentations.transforms.RandomBrightnessContrast',
                'params': {
                    'brightness_by_max': True,
                    'brightness_limit': (-0.2, 0.2),
                    'contrast_limit': (-0.1, 0.1),
                    'p': 0.5,
                    'always_apply': True
                }
            },
            {
                'name': 'albumentations.augmentations.transforms.RandomGridShuffle',
                'params': {'grid': [4, 4], 'p':0.6}
            }

        ]

        # Check size of loaded transforms
        assert len(expected_transforms) == len(transforms), \
            f"Size mismatch! Expected transforms = {len(expected_transforms) }, Loaded transform = {len(transforms)}"

        # Check each transform name/arguments
        for idx, exp in enumerate(expected_transforms):
            t = transforms[idx]
            # print(t)
            assert t[name_field] == exp['name'], f"Transform name {t[name_field]} is wrong! Expected: {exp['name']}"
            for param, value in exp['params'].items():
                assert param in t, f"Param {param} not found in transform {t}"
                assert t[param] == value, f"{exp['name']}: Param {param}={t[param]} is wrong! Expected: {value}"

    def test_load_invalid_configuration_sample(self, augmentations_folder):

        cfg_file = augmentations_folder / 'invalid_augmentations.yml'

        with pytest.raises(ModuleNotFoundError):
            composition = TransformsFactory.parse_file(cfg_file, raise_not_found_error=True)

        composition = TransformsFactory.parse_file(cfg_file, raise_not_found_error=False)
        composition_dict = composition.get_dict_with_id()

        # Check for composition consistency
        assert 'transforms' in composition_dict, "composition dict is wrong!"
        transforms = composition_dict['transforms']

        # Check transforms are empty!
        assert len(transforms) == 0, "Strange transforms was loaded despite wrong names!"

    def test_interpolations_and_borders(self, augmentations_folder):

        cfg_file = augmentations_folder / 'interpolations_and_borders.yml'

        composition = TransformsFactory.parse_file(cfg_file, raise_not_found_error=False)
        composition_dict = composition.get_dict_with_id()

        # Check for composition consistency
        assert 'transforms' in composition_dict, "composition dict is wrong!"
        transforms = composition_dict['transforms']

        name_field = '__class_fullname__'

        # Creates Interpolation/Border Pairs
        interpolations_and_borders = [
            ('linear', 'constant'),
            ('none', 'reflect'),
            ('cubic', 'reflect101'),
            ('default', 'replicate'),
            ('default', 'wrap'),
            ('default', 'default'),
        ]

        # Create Expted Transforms based on Pairs
        expected_transforms = []
        for interpolation, border in interpolations_and_borders:
            expected_transforms.append(
                {
                    'name': 'albumentations.augmentations.transforms.Rotate',
                    'params': {
                        'interpolation': AlbumentationTransformsFactory.INTERPOLATIONS[interpolation],
                        'border_mode': AlbumentationTransformsFactory.BORDERS[border],
                    }
                })

        # Check size of loaded transforms
        assert len(expected_transforms) == len(transforms), \
            f"Size mismatch! Expected transforms = {len(expected_transforms) }, Loaded transform = {len(transforms)}"

        # Check each transform name/arguments
        for idx, exp in enumerate(expected_transforms):
            t = transforms[idx]
            # print(t)
            assert t[name_field] == exp['name'], f"Transform name {t[name_field]} is wrong! Expected: {exp['name']}"
            for param, value in exp['params'].items():
                assert param in t, f"T{idx}: Param {param} not found in transform {t}"
                assert t[param] == value, f"T{idx}: {exp['name']}: Param {param}={t[param]} is wrong! Expected: {value}"
