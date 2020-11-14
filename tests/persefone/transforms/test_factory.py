import warnings
import pytest
from persefone.transforms.factory import TransformsFactory, AlbumentationTransformsFactory
import numpy as np
import torch


@pytest.mark.transforms
class TestTransformsFactory(object):

    @pytest.fixture(scope='session')
    def expected_transforms(self):
        return [
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
            },
            {
                'name': 'albumentations.augmentations.transforms.HueSaturationValue',
                'params': {
                    'hue_shift_limit': (0.1, 0.2),
                    'sat_shift_limit': (0.3, 0.4),
                    'val_shift_limit': (0.5, 0.6),
                    'p': 0.7,
                    'always_apply': True
                }
            },
            {
                'name': 'albumentations.augmentations.transforms.HorizontalFlip',
                'params': {'p': 0.8, 'always_apply': False}
            },
            {
                'name': 'albumentations.augmentations.transforms.VerticalFlip',
                'params': {'p': 0.9, 'always_apply': False}
            },
            {
                'name': 'albumentations.augmentations.transforms.Flip',
                'params': {'p': 0.10, 'always_apply': False}
            },
            {
                'name': 'albumentations.augmentations.transforms.ShiftScaleRotate',
                'params': {
                    'shift_limit_x': [-0.1, 0.1],
                    'shift_limit_y': [-0.2, 0.2],
                    'scale_limit': (0.3, 0.4),
                    'rotate_limit': (-22, 22),
                    'interpolation': AlbumentationTransformsFactory.INTERPOLATIONS['linear'],
                    'border_mode': AlbumentationTransformsFactory.BORDERS['replicate'],
                    'value': [200, 100, 90],
                    'mask_value': 155,
                    'p': 0.10,
                    'always_apply': False
                }
            },
            {
                'name': 'albumentations.augmentations.transforms.Normalize',
                'params': {
                    'mean': (0.5, 0.5, 0.3),
                    'std': (0.1, 0.1, 0.1),
                    'always_apply': False,
                    'p': 0.4
                }
            },
            {
                'name': 'albumentations.augmentations.transforms.CoarseDropout',
                'params': {
                    'max_holes': 2,
                    'min_holes': 1,
                    'max_width': 64,
                    'max_height': 128,
                    'min_width': 12,
                    'min_height': 53,
                    'fill_value': 10,
                    'always_apply': True,
                    'p': 0.8,
                }
            },
            {
                'name': 'albumentations.augmentations.transforms.PadIfNeeded',
                'params': {
                    'min_height': 1000,
                    'min_width': 1000,
                    'border_mode': AlbumentationTransformsFactory.BORDERS['replicate'],
                    'value': [0, 0, 0],
                    'mask_value': 0,
                    'always_apply': True,
                    'p': 0.8,
                }
            },
            {
                'name': 'persefone.transforms.custom.random_stain.RandomStain',
                'params': {
                    'min_holes': 4,
                    'max_holes': 15,
                    'min_size': 32,
                    'max_size': 32,
                    'min_eccentricity': 1,
                    'max_eccentricity': 3,
                    'fill_mode': 'solid',
                    'min_rgb': [0.5, 0.5, 0.5],
                    'max_rgb': [1.0, 1.0, 1.0],
                    'n_points': 20,
                    'perturbation_radius': 10,
                    'noise': 10,
                    'always_apply': True,
                    'p': 0.5,
                }
            },
            {
                'name': 'persefone.transforms.custom.pad_if_needed_v2.PadIfNeededV2',
                'params': {
                    'min_height': 1024,
                    'min_width': 512,
                    'border_mode': AlbumentationTransformsFactory.BORDERS['replicate'],
                    'value': 10,
                    'mask_value': 0,
                    'row_align': "bottom",
                    'col_align': "left",
                    'always_apply': True,
                    'p': 0.8
                }
            },
            {
                'name': 'albumentations.augmentations.transforms.LongestMaxSize',
                'params': {
                    'max_size': 1024,
                    'p': 0.44,
                    'always_apply': True,
                    'interpolation': AlbumentationTransformsFactory.INTERPOLATIONS['linear'],
                }
            },
            {
                'name': 'albumentations.augmentations.transforms.SmallestMaxSize',
                'params': {
                    'max_size': 1024,
                    'p': 0.33,
                    'always_apply': True,
                    'interpolation': AlbumentationTransformsFactory.INTERPOLATIONS['linear'],
                }
            },
            {
                'name': 'albumentations.augmentations.transforms.CenterCrop',
                'params': {
                    'height': 50,
                    'width': 100,
                    'p': 0.22,
                    'always_apply': False
                }
            },
            {
                'name': 'albumentations.imgaug.transforms.IAAPerspective',
                'params': {
                    'scale': [0.06, 0.12],
                    'keep_size': True,
                    'p': 0.11,
                    'always_apply': False
                }
            },
            {
                'name': 'albumentations.augmentations.transforms.Blur',
                'params': {
                    'blur_limit': [7, 8],
                    'p': 0.05,
                    'always_apply': False
                }
            },
            {
                'name': 'albumentations.augmentations.transforms.InvertImg',
                'params': {
                    'p': 0.05,
                    'always_apply': False
                }
            },
            {
                'name': 'albumentations.augmentations.transforms.GaussNoise',
                'params': {
                    'var_limit': [5., 7.],
                    'p': 0.05,
                    'always_apply': False
                }
            }


        ]

    def _compare_param(self, p1, p2):
        if isinstance(p1, list) or isinstance(p1, tuple):
            return np.all(np.isclose(np.array(p1), np.array(p2)))
        else:
            return p1 == p2

    def test_load_full_configuration_sample(self, augmentations_folder, expected_transforms):

        cfg_file = augmentations_folder / 'full_augmentations.yml'
        composition = TransformsFactory.parse_file(cfg_file)
        composition_dict = composition.get_dict_with_id()

        # Check for composition consistency
        assert 'transforms' in composition_dict, "composition dict is wrong!"
        transforms = composition_dict['transforms']

        # Creates Expected Transforms List. Has to match with YAML Configuration file list
        name_field = '__class_fullname__'

        # Check size of loaded transforms
        assert len(expected_transforms) == len(transforms), \
            f"Size mismatch! Expected transforms = {len(expected_transforms) }, Loaded transform = {len(transforms)}"

        # Check each transform name/arguments
        for idx, exp in enumerate(expected_transforms):
            t = transforms[idx]
            # print(t)
            # TODO: fill_value is not retrivable with t.get_dict_with_id()
            if exp['name'] == 'albumentations.augmentations.transforms.CoarseDropout':
                warnings.warn('param fill_value of CoarseDropout is not retrievable with get_dict_with_id -- SKIPPING THIS TRANSFORM')
                continue
            assert t[name_field] == exp['name'], f"Transform name {t[name_field]} is wrong! Expected: {exp['name']}"
            for param, value in exp['params'].items():
                assert param in t, f"Param {param} not found in transform {t}"
                assert self._compare_param(t[param], value), f"{exp['name']}: Param {param}={t[param]} is wrong! Expected: {value}"

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


@pytest.mark.transforms
class TestPytorchTransformsFactory(object):

    @pytest.fixture(scope='session')
    def expected_transforms(self):

        return [
            {
                'name': 'persefone.transforms.pytorch.ToTensorExtended',
                'params': {
                    'in_type': np.uint8,
                    'in_range': [0, 255],
                    'out_type': torch.float32,
                    'out_range': [0, 1.],
                    'always_apply': True,
                    'p': 0.4
                }
            },
            {
                'name': 'persefone.transforms.pytorch.ToTensorExtended',
                'params': {
                    'in_type': np.float32,
                    'in_range': [0., 1.],
                    'out_type': torch.float64,
                    'out_range': [-1, 1.],
                    'always_apply': True,
                    'p': 0.5
                }
            },
            {
                'name': 'persefone.transforms.pytorch.ToTensorExtended',
                'params': {
                    'in_type': np.uint8,
                    'in_range': [0, 255],
                    'out_type': torch.int16,
                    'out_range': [0, 1000],
                    'always_apply': True,
                    'p': 0.3
                }
            },
        ]

    def _compare_param(self, p1, p2):
        if isinstance(p1, list) or isinstance(p1, tuple):
            return np.all(np.isclose(np.array(p1), np.array(p2)))
        else:
            return p1 == p2

    def test_load_full_configuration_sample(self, augmentations_folder, expected_transforms):

        cfg_file = augmentations_folder / 'pytorch_augmentations.yml'
        composition = TransformsFactory.parse_file(cfg_file)
        composition_dict = composition.get_dict_with_id()

        # Check for composition consistency
        assert 'transforms' in composition_dict, "composition dict is wrong!"
        transforms = composition_dict['transforms']

        # Creates Expected Transforms List. Has to match with YAML Configuration file list
        name_field = '__class_fullname__'

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
                assert self._compare_param(t[param], value), f"{exp['name']}: Param {param}={t[param]} is wrong! Expected: {value}"

    def test_data_consistency(self, augmentations_folder, expected_transforms):

        cfg_file = augmentations_folder / 'pytorch_augmentations.yml'
        composition = TransformsFactory.parse_file(cfg_file)

        size = [256, 256, 3]
        for idx, exp in enumerate(expected_transforms):
            print(exp)
            in_range = exp['params']['in_range']
            in_type = exp['params']['in_type']
            out_range = exp['params']['out_range']
            out_type = exp['params']['out_type']
            print("RANGE", in_range, in_type)

            fake_image = np.random.uniform(low=in_range[0], high=in_range[1], size=size).astype(in_type)
            transform = composition[idx]
            t_image = transform(image=fake_image)['image']
            assert isinstance(t_image, torch.Tensor), "Output is not a pytorch.Tensor!"
            assert t_image.dtype == out_type, "Output type is wrong!"
            assert t_image.max() <= out_range[1], "Output max is greater thant max range"
            assert t_image.min() >= out_range[0], "Output max is greater thant max range"
            print("MIN MAX", t_image.min(), t_image.max())

        # TODO: a numeric consistency test is missing here. No check if out MAX/MIN is correct

    def test_empty_transforms(self, augmentations_folder, expected_transforms):

        cfg_file = augmentations_folder / 'pytorch_augmentations_empty.yml'
        composition = TransformsFactory.parse_file(cfg_file)
        print(composition)
