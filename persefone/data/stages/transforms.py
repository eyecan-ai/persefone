
from pathlib import Path

import albumentations
from persefone.transforms.factory import TransformsFactory
from typing import Dict, Union
from persefone.data.databases.filesystem.underfolder import UnderfolderDatabase
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
