from logging import debug
from pathlib import Path

import yaml
from persefone.data.databases.filesystem.underfolder import (
    UnderfolderDatabase
)
import numpy as np
from persefone.data.stages.base import (
    StagesComposition, StageQuery, StageGroupBy, StageKeyFiltering, StageSubsampling
)
from persefone.data.stages.transforms import StageTransforms


class TestStages(object):

    def test_simple_stages(self, underfoldertomix_folder, augmentations_folder):

        augmentation_file = underfoldertomix_folder / 'augmentations.yml'
        datasets = [
            UnderfolderDatabase(folder=underfoldertomix_folder),
            UnderfolderDatabase(folder=underfoldertomix_folder, use_lazy_samples=True)
        ]

        for dataset in datasets:

            stages = StagesComposition([
                StageSubsampling(factor=2),
                StageKeyFiltering(['image', 'metadata']),
                StageQuery(queries=[
                    '`metadata.counter` > 0',
                ], debug=True),
                StageQuery(queries=[
                    '`metadata.counter` > 0',
                ], debug=False),
                StageKeyFiltering({
                    'metadata': 'meta',
                    'image': 'x'
                }),
                StageTransforms(augmentations=augmentation_file),
                StageTransforms(augmentations=yaml.safe_load(open(augmentation_file, 'r')))
            ])

            staged_dataset = stages(dataset)
            assert len(staged_dataset) > 0
            print(len(dataset), len(staged_dataset))

    def test_groupby_stage_stacked(self, underfoldertomix_folder, augmentations_folder):

        augmentation_file = underfoldertomix_folder / 'augmentations.yml'
        datasets = [
            UnderfolderDatabase(folder=underfoldertomix_folder),
            UnderfolderDatabase(folder=underfoldertomix_folder, use_lazy_samples=True)
        ]

        for dataset in datasets:

            stages = StagesComposition([
                StageGroupBy('metadata.counter', stack_values=True)
            ])

            staged_dataset = stages(dataset)
            assert len(staged_dataset) < len(dataset)

            shapes_map = {}
            for sample_id in range(len(dataset)):
                sample = dataset[sample_id]
                for key in sample.keys():
                    if isinstance(sample[key], np.ndarray):
                        shapes_map[key] = sample[key].shape
                    else:
                        shapes_map[key] = (1,)
                break

            for sample_id in range(len(staged_dataset)):
                sample = staged_dataset[sample_id]
                for key in sample.keys():
                    if isinstance(sample[key], np.ndarray):
                        if len(sample[key].shape) > 1:
                            assert len(sample[key].shape) > len(shapes_map[key]), f'{key} shape'
                        else:
                            assert sample[key].shape[0] > shapes_map[key][0], f'{key} shape'

    def test_groupby_stage_unstacked(self, underfoldertomix_folder, augmentations_folder):

        augmentation_file = underfoldertomix_folder / 'augmentations.yml'
        datasets = [
            UnderfolderDatabase(folder=underfoldertomix_folder),
            UnderfolderDatabase(folder=underfoldertomix_folder, use_lazy_samples=True)
        ]

        for dataset in datasets:

            stages = StagesComposition([
                StageGroupBy('metadata.counter', stack_values=False)
            ])

            staged_dataset = stages(dataset)
            assert len(staged_dataset) < len(dataset)

            keys = []
            for sample_id in range(len(dataset)):
                sample = dataset[sample_id]
                keys.extend(list(sample.keys()))
                break

            for sample_id in range(len(staged_dataset)):
                sample = staged_dataset[sample_id]
                assert len(sample.keys()) > len(keys)

    def test_groupby_stage_wrong(self, underfoldertomix_folder, augmentations_folder):

        augmentation_file = underfoldertomix_folder / 'augmentations.yml'
        dataset = UnderfolderDatabase(folder=underfoldertomix_folder)

        StageGroupBy('metadata.XXX', stack_values=False)(dataset)
        StageGroupBy('metadata.XXX', stack_values=False, debug=True)(dataset)
