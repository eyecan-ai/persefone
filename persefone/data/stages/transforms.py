
from pathlib import Path
from persefone.transforms.factory import TransformsFactory
from typing import Dict, Union
from persefone.data.databases.filesystem.underfolder import UnderfolderDatabase
from persefone.data.stages.base import DStage
import yaml


class StageTransforms(DStage):

    def __init__(self, augmentations: Union[dict, str]):
        """ Stage for augmentations/transforms

        :param augmentations: augmentation configuration or filename. @see TransformsFactory
        :type augmentations: Union[dict, str]
        :raises NotImplementedError: error for invalid augmentations type
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
        self._tramsforms_targets = self._transforms_cfg.get('inputs', {})
        self._transforms.add_targets(self._tramsforms_targets)

    def __getitem__(self, idx):

        if idx >= len(self):
            raise IndexError

        sample = dict(self._dataset[idx])

        # Creates map of input data
        to_transform = {}
        for key in self._tramsforms_targets.keys():
            if key in sample:
                to_transform[key] = sample[key]

        # Applies transforms to targets
        transformed = self._transforms(**to_transform)

        # Replace original fields
        for key in self._tramsforms_targets.keys():
            if key in sample:
                sample[key] = transformed[key]

        return sample
