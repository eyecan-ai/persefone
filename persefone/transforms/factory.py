import yaml
from persefone.utils.pyutils import get_arg
import albumentations as A
from .albumentations import AlbumentationTransformsFactory
from .pytorch import PytorchTransformationsFactory
from .custom.custom import CustomTransformsFactory
import logging

_logger = logging.getLogger()


class TransformsFactory(object):

    @classmethod
    def _factories_map(cls):
        return {
            'aug': AlbumentationTransformsFactory,
            'pytorch': PytorchTransformationsFactory,
            'custom': CustomTransformsFactory,
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
    def parse_dict(cls, cfg: dict, raise_not_found_error: bool = False, ignore_params=False) -> A.Compose:
        """ Parse dictionay into Transforms Compositions

        :param cfg: input configuration dictionary
        :type cfg: dict
        :param raise_not_found_error: TRUE to raise exception for factory miss detection, defaults to False
        :type raise_not_found_error: bool, optional
        :param ignore_params: TRUE to ignore stored params infor for Bboxes or Keypoints, defaults to False
        :type ignore_params: bool, optional
        :raises ModuleNotFoundError: raise exception if transform item is not recognized
        :return: Transforms composition
        :rtype: A.Compose
        """

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

        if not ignore_params:
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

    @classmethod
    def parse_file(cls, filename: str, raise_not_found_error: bool = False, ignore_params: bool = False) -> A.Compose:
        """ Parse configuration file into Transforms Compositions

        :param filename: configuration filename
        :type filename: str
        :param raise_not_found_error: TRUE to raise exception for factory miss detection, defaults to False
        :type raise_not_found_error: bool, optional
        :param ignore_params: TRUE to ignore stored params infor for Bboxes or Keypoints, defaults to False
        :type ignore_params: bool, optional
        :raises ModuleNotFoundError: raise exception if transform item is not recognized
        :return: Transforms composition
        :rtype: A.Compose
        """

        cfg = yaml.safe_load(open(filename))
        return cls.parse_dict(cfg, raise_not_found_error=raise_not_found_error, ignore_params=ignore_params)
