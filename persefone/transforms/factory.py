import yaml
import albumentations as A
from persefone.utils.pyutils import get_arg
import cv2
import logging


logging.basicConfig(level=logging.DEBUG)
_logger = logging.getLogger()


class AlbumentationTransformsFactory(object):

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
    def _targets_map(cls):
        return {
            'pixels': ['image', 'mask', 'masks', 'bboxes', 'keypoints'],
            'spatial_full': ['image', 'mask', 'masks', 'bboxes', 'keypoints'],
            'spatial_half': ['image', 'mask', 'masks'],
        }

    @classmethod
    def _functions_map(cls):
        return {
            'resize': {'f': cls._build_resize_transform, 'targets': cls._targets_map()['spatial_full']},
            'rotate': {'f': cls._build_rotate_transform, 'targets': cls._targets_map()['spatial_full']},
            'crop': {'f': cls._build_crop_transform, 'targets': cls._targets_map()['spatial_full']},
            'random_crop': {'f': cls._build_random_crop_transform, 'targets': cls._targets_map()['spatial_full']},
            'random_brightness_contrast': {'f': cls._build_random_brightness_contrast, 'targets': cls._targets_map()['pixels']},
            'random_grid_shuffle': {'f': cls._build_random_grid_shuffle, 'targets': cls._targets_map()['spatial_half']},
        }

    @classmethod
    def _build_resize_transform(cls, **params):
        size = get_arg(params, 'size', [256, 256])
        p = get_arg(params, 'p', 1.0)
        always = get_arg(params, 'always', True)
        interpolation = cls._get_interpolation_value(get_arg(params, 'interpolation', 'none'))
        return A.Resize(
            height=size[1],
            width=size[0],
            interpolation=interpolation,
            always_apply=always,
            p=p
        )

    @classmethod
    def _build_rotate_transform(cls, **params):
        return A.Rotate(
            limit=get_arg(params, 'limit', [-180, 180]),
            interpolation=cls._get_interpolation_value(get_arg(params, 'interpolation', 'none')),
            border_mode=cls._get_borders_value(get_arg(params, 'border_mode', 'constant')),
            value=get_arg(params, 'value', 0),
            mask_value=get_arg(params, 'mask_value', 0),
            always_apply=get_arg(params, 'always_apply', False),
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
            always_apply=get_arg(params, 'always_apply', False),
            p=get_arg(params, 'p', 1.0)
        )

    @classmethod
    def _build_random_crop_transform(cls, **params):
        size = get_arg(params, 'size', [50, 50])
        return A.RandomCrop(
            height=size[1],
            width=size[0],
            always_apply=get_arg(params, 'always_apply', False),
            p=get_arg(params, 'p', 1.0)
        )

    @classmethod
    def _build_random_brightness_contrast(cls, **params):
        return A.RandomBrightnessContrast(
            brightness_limit=get_arg(params, 'brightness', 0.2),
            contrast_limit=get_arg(params, 'contrast', 0.1),
            brightness_by_max=get_arg(params, 'brighntess_by_max', True),
            always_apply=get_arg(params, 'always_apply', False),
            p=get_arg(params, 'p', 1.0)
        )

    @classmethod
    def _build_random_grid_shuffle(cls, **params):
        return A.RandomGridShuffle(
            grid=get_arg(params, 'grid', [3, 3]),
            always_apply=get_arg(params, 'always_apply', False),
            p=get_arg(params, 'p', 1.0)
        )

    @classmethod
    def build_transform(cls, name, **params):
        functions_map = cls._functions_map()
        if name not in functions_map:
            _logger.error(f'Transform: {name} not found in {cls.__name__}')
            return None, []
        return functions_map[name]['f'](**params), functions_map[name]['targets']


class TransformsFactory(object):

    @classmethod
    def _factories_map(cls):
        return {
            'aug': AlbumentationTransformsFactory
        }

    @classmethod
    def _build_bboxes_configuration(cls, **params):
        return A.BboxParams(
            format=get_arg(params, 'type', 'coco'),
            label_fields=get_arg(params, 'label_fields', ['category_id']),
            min_area=get_arg(params, 'min_area', 0.0),
            min_visibility=get_arg(params, 'min_visibility', 0.0)
        )

    @classmethod
    def _build_keypoints_configuration(cls, **params):
        return A.KeypointParams(
            format=get_arg(params, 'format', 'xy'),
            label_fields=get_arg(params, 'label_fields', ['category_id']),
            remove_invisible=get_arg(params, 'remove_invisible', True),
            angle_in_degrees=get_arg(params, 'angle_in_degrees', True)
        )

    @classmethod
    def _parse_item(cls, item: dict):
        assert len(item.keys()) == 1
        item_name = list(item.keys())[0]
        if '.' in item_name:
            namespace, name = item_name.split('.')
        else:
            namespace, name = '', item_name

        if namespace not in cls._factories_map():
            logging.error(f'{cls.__name__}._parse_item: namespace "{namespace}" not found!')
            return None, []

        factory: AlbumentationTransformsFactory = cls._factories_map()[namespace]

        transform, targets = factory.build_transform(name, **item[item_name])
        return transform, targets

    @classmethod
    def _parse_transforms(cls, cfg):
        transforms_items = []
        if 'transforms' in cfg:
            transforms_items = cfg['transforms']

        transforms_with_targets = []
        for transform_item in transforms_items:
            transform, targets = cls._parse_item(transform_item)
            transforms_with_targets.append((transform, targets))

        return transforms_with_targets

    @classmethod
    def _parse_inputs(cls, cfg):

        inputs = {'image': 'image'}

        if 'inputs' in cfg:
            _inputs = cfg['inputs']
            if isinstance(_inputs, dict):
                inputs = _inputs

        return inputs

    @classmethod
    def parse_file(cls, filename, raise_not_found_error=False):
        cfg = yaml.safe_load(open(filename))

        inputs = cls._parse_inputs(cfg)
        inputs_set = set(inputs.keys())

        transforms_with_targets = cls._parse_transforms(cfg)
        transforms = []
        valids = []
        for transform, targets in transforms_with_targets:
            if transform is not None:
                targets_set = set(targets)
                valids.append(inputs_set.issubset(targets_set))
                transforms.append(transform)
            else:
                if raise_not_found_error:
                    raise ModuleNotFoundError("")

        bboxes_params = None
        keypoints_params = None

        if 'bboxes' in inputs:
            bboxes_params = cls._build_bboxes_configuration(**inputs['bboxes'])

        if 'keypoints' in inputs:
            keypoints_params = cls._build_keypoints_configuration(**inputs['keypoints'])

        composition = A.Compose(
            transforms=transforms,
            bbox_params=bboxes_params,
            keypoint_params=keypoints_params
        )

        return composition
