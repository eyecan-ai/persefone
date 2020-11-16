
from pathlib import Path

import numpy as np
from persefone.transforms.factory import TransformsFactory
from typing import Any, Dict, Sequence, Union
from persefone.data.stages.base import DStage
import yaml


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

    def __init__(self, range_remap: Dict[str, Dict[str, Any]]) -> None:
        super().__init__()
        self._range_remap = range_remap

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
            a = (o_max - o_min) / (i_max - i_min)
            sample[key] = ((sample[key] - i_min) * a + o_min).astype(dtype)

        return sample
