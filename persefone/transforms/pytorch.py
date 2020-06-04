from .base import AbstractFactory
from persefone.utils.pyutils import get_arg
import logging
from albumentations.core.transforms_interface import BasicTransform
import torch
import numpy as np

_logger = logging.getLogger()


class ToTensorExtended(BasicTransform):
    """Convert image and mask to `torch.Tensor`."""

    def __init__(self, in_type=np.uint8, in_range=[0, 255], out_type=torch.float32, out_range=[0., 1.], always_apply=True, p=1.0):
        super(ToTensorExtended, self).__init__(always_apply=always_apply, p=p)
        self.in_type = eval(in_type) if isinstance(in_type, str) else in_type
        self.out_type = eval(out_type) if isinstance(out_type, str) else out_type
        self.in_range = in_range
        self.out_range = out_range

    @property
    def targets(self):
        return {'image': self.apply, 'mask': self.apply_to_mask}

    def apply(self, img, **params):  # skipcq: PYL-W0613
        img = np.interp(
            img,
            (self.in_range[0], self.in_range[1]),
            (self.out_range[0], self.out_range[1])
        )
        img = torch.from_numpy(img.transpose(2, 0, 1))
        img = img.type(self.out_type)
        return img

    def apply_to_mask(self, mask, **params):  # skipcq: PYL-W0613
        mask = np.interp(
            mask,
            (self.in_range[0], self.in_range[1]),
            (self.out_range[0], self.out_range[1])
        )
        mask = torch.from_numpy(mask)
        mask = mask.type(self.out_type)
        return mask

    def get_transform_init_args_names(self):
        return ['in_type', 'in_range', 'out_type', 'out_range']

    def get_params_dependent_on_targets(self, params):
        return {}


class PytorchTransformationsFactory(AbstractFactory):

    @classmethod
    def _build_to_tensor_transform(cls, **params):
        return ToTensorExtended(
            in_type=get_arg(params, 'in_type', np.uint8),
            out_type=get_arg(params, 'out_type', torch.float32),
            in_range=get_arg(params, 'in_range', [0, 255]),
            out_range=get_arg(params, 'out_range', [0., 1, ]),
            always_apply=get_arg(params, 'always_apply', True),
            p=get_arg(params, 'p', 1.0)
        )

    @classmethod
    def _functions_map(cls):
        return {
            'to_tensor': {'f': cls._build_to_tensor_transform, 'targets': cls._targets_map()['spatial_full']},
        }

    @classmethod
    def build_transform(cls, name, **params):
        functions_map = cls._functions_map()
        if name not in functions_map:
            _logger.error(f'Pytorch Transform: {name} not found in {cls.__name__}')
            return None, []
        return functions_map[name]['f'](**params), functions_map[name]['targets']
