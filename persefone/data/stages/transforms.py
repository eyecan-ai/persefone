
from typing import Any, Dict, Iterable, Sequence, Union
from pathlib import Path

import numpy as np
import yaml
from schema import Or, Schema, And, Use, Optional

from persefone.transforms.factory import TransformsFactory
from persefone.data.stages.base import DStage, StagesComposition


class StageTransforms(DStage):

    NEEDED_IMAGE_KEY = 'image'

    def __init__(self, augmentations: Union[dict, str], default_image_key: str = 'image'):
        """ Stage for augmentations/transforms

        :param augmentations: augmentation configuration or filename. @see TransformsFactory
        :type augmentations: Union[dict, str]
        :raises NotImplementedError: error for invalid augmentations type
        :param default_image_key: Default first element key for albumentations transform. usually is 'image'
        :type default_image_key: str
        """
        super().__init__()

        self._transforms_cfg = {}
        if isinstance(augmentations, str) or isinstance(augmentations, Path):
            self._transforms_cfg = yaml.safe_load(open(augmentations, 'r'))
        elif isinstance(augmentations, dict):
            self._transforms_cfg = augmentations
        else:
            raise NotImplementedError(f'Invalid augmentations type [{augmentations}]')

        self._transforms = TransformsFactory.parse_dict(self._transforms_cfg)
        self._transforms_targets = self._transforms_cfg.get('inputs', {})
        self._transforms.add_targets(self._transforms_targets)
        self._default_image_key = default_image_key

    def __getitem__(self, idx):

        if idx >= len(self):
            raise IndexError

        sample = dict(self._dataset[idx])

        # Creates map of input data
        to_transform = {}
        for key in self._transforms_targets.keys():
            if key in sample:
                to_transform[key] = sample[key]

        assert self._default_image_key in to_transform, f"Missing default '{self._default_image_key}' key in samples"

        if self.NEEDED_IMAGE_KEY != self._default_image_key:
            to_transform[self.NEEDED_IMAGE_KEY] = to_transform[self._default_image_key]
            del to_transform[self._default_image_key]

        # Applies transforms to targets
        transformed = self._transforms(**to_transform)
        if self.NEEDED_IMAGE_KEY != self._default_image_key:
            transformed[self._default_image_key] = transformed[self.NEEDED_IMAGE_KEY]

        # Replace original fields
        for key in self._transforms_targets.keys():
            if key in sample:
                sample[key] = transformed[key]

        return sample


class StageTranspose(DStage):
    """Transposes axis in specified numpy arrays

    :param transposition: transposition operation parameters. Must contain,
    for each dataset key to transpose, a sequence of integers representing the axis to transpose
    :type transposition: Dict[str, Sequence[int]]
    """

    def __init__(self, transposition: Dict[str, Sequence[int]]) -> None:
        super().__init__()
        self._transposition = transposition

    def __getitem__(self, idx) -> dict:
        if idx >= len(self):
            raise IndexError

        sample = dict(self._dataset[idx])

        # Applies transposition to target keys
        for key in self._transposition:
            sample[key] = np.transpose(sample[key], self._transposition[key])

        return sample


class StageRangeRemap(DStage):
    """Remaps all values from input range to specified output range, then casts numpy arrays to
    desired dtype.

    :param range_remap: range remap operation parameters. Must contain,
    for each dataset key to remap, a dictionary with the following keys:
    `in_range` - input data range, tuple or list of two integers or floats
    `out_range` - output data range, tuple or list of two integers or floats
    `dtype` - output dtype
    `clamp` - optional, if True clamps input data to its range,
    else expects input data to be already in range and raises AssertionError if it is not, default False
    :type range_remap: Dict[str, Dict[str, Any]]
    """

    range_remap_schema = Schema({
        str: {
            'in_range': And(Use(list), Or([int, int], [float, float])),
            'out_range': And(Use(list), Or([int, int], [float, float])),
            'dtype': str,
            Optional('clamp', default=False): bool
        }
    })

    def __init__(self, range_remap: Dict[str, Dict[str, Any]]) -> None:
        super().__init__()
        self._range_remap = self.range_remap_schema.validate(range_remap)

    def __getitem__(self, idx) -> dict:
        if idx >= len(self):
            raise IndexError

        sample = dict(self._dataset[idx])

        # Applies range remap to target keys
        for key in self._range_remap:
            i_min = self._range_remap[key]['in_range'][0]
            i_max = self._range_remap[key]['in_range'][1]
            o_min = self._range_remap[key]['out_range'][0]
            o_max = self._range_remap[key]['out_range'][1]
            dtype = self._range_remap[key]['dtype']
            clamp = self._range_remap[key]['clamp']

            if clamp:
                sample[key] = np.clip(sample[key], i_min, i_max)
            else:
                assert np.max(sample[key]) <= i_max
                assert np.min(sample[key]) >= i_min

            a = (o_max - o_min) / (i_max - i_min)
            sample[key] = ((sample[key] - i_min) * a + o_min).astype(dtype)

        return sample


class StageToCHWFloat(StagesComposition):
    """Transforms input images from "numpy" format ([H, W, C], uint8 [0, 255]) to "torch" format ([C, H, W], float [0., 1.])
    Output images will still be numpy arrays.

    :param keys: iterable of keys to which apply the transformation
    :type keys: Iterable[str]
    """

    def __init__(self, keys: Iterable[str]) -> None:
        transposition = (1, 2, 0)
        params = {
            'in_range': (0, 255),
            'out_range': (0., 1.),
            'dtype': 'float'
        }
        super().__init__([
            StageTranspose({k: transposition for k in keys}),
            StageRangeRemap({k: params for k in keys})
        ])


class StageToHWCUint8(StagesComposition):
    """Transforms input images from "torch" format ([C, H, W], float [0., 1.]) to "numpy" format ([H, W, C], uint8 [0, 255])
    Input images must be numpy arrays.

    :param keys: iterable of keys to which apply the transformation
    :type keys: Iterable[str]
    """

    def __init__(self, keys: Iterable[str]) -> None:
        transposition = (2, 0, 1)
        params = {
            'in_range': (0., 1.),
            'out_range': (0, 255),
            'dtype': 'uint8'
        }
        super().__init__([
            StageTranspose({k: transposition for k in keys}),
            StageRangeRemap({k: params for k in keys})
        ])
