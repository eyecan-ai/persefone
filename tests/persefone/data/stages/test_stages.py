import yaml
import numpy as np
import pytest
from persefone.data.stages.base import (
    StageCache, StagesComposition, StageQuery, StageGroupBy, StageKeyFiltering, StageSubsampling
)
from persefone.data.stages.transforms import StageToCHWFloat, StageToHWCUint8, StageTransforms, StageTranspose, StageRangeRemap
from persefone.data.databases.filesystem.underfolder import (
    UnderfolderDatabase
)


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

        dataset = UnderfolderDatabase(folder=underfoldertomix_folder)

        StageGroupBy('metadata.XXX', stack_values=False)(dataset)
        StageGroupBy('metadata.XXX', stack_values=False, debug=True)(dataset)

    def test_stage_transpose(self, underfolder_folder):
        datasets = [
            UnderfolderDatabase(folder=underfolder_folder),
            UnderfolderDatabase(folder=underfolder_folder, use_lazy_samples=True)
        ]

        for dataset in datasets:
            transposition = {
                'image': (2, 0, 1),
                'image_mask': (0, 1, 2),
            }

            stage = StageTranspose(transposition)

            staged_dataset = stage(dataset)
            sample = dataset[0]
            staged_sample = staged_dataset[0]
            expected = {
                k: np.transpose(sample[k], v)
                for k, v in transposition.items()
            }
            assert sample.keys() == staged_sample.keys()
            for k in transposition:
                assert np.allclose(expected[k], staged_sample[k])

    def test_stage_range_remap(self, underfolder_folder):
        datasets = [
            UnderfolderDatabase(folder=underfolder_folder),
            UnderfolderDatabase(folder=underfolder_folder, use_lazy_samples=True)
        ]

        for dataset in datasets:
            for clamp in [True, False]:
                range_remap = {
                    'image': {
                        'in_range': (0, 255),
                        'out_range': (0., 1.),
                        'dtype': 'float',
                        'clamp': clamp
                    },
                }
                stage = StageRangeRemap(range_remap)
                staged_dataset = stage(dataset)
                sample = dataset[0]
                staged_sample = staged_dataset[0]

                expected = {}
                for k, v in range_remap.items():
                    a = (v['out_range'][1] - v['out_range'][0]) / (v['in_range'][1] - v['in_range'][0])
                    expected[k] = (sample[k] - v['in_range'][0]) * a + v['out_range'][0]
                    expected[k] = expected[k].astype(v['dtype'])

                assert sample.keys() == staged_sample.keys()
                for k in range_remap:
                    assert np.allclose(expected[k], staged_sample[k])

            # Check raises when input is outside of specified range
            range_remap = {
                'image': {
                    'in_range': (-100, -50),
                    'out_range': (0., 1.),
                    'dtype': 'float',
                    'clamp': False
                },
            }
            stage = StageRangeRemap(range_remap)
            staged_dataset = stage(dataset)
            with pytest.raises(AssertionError):
                staged_dataset[0]

    def test_stage_to_chw_float(self, underfolder_folder):
        datasets = [
            UnderfolderDatabase(folder=underfolder_folder),
            UnderfolderDatabase(folder=underfolder_folder, use_lazy_samples=True)
        ]

        for dataset in datasets:
            keys = ['image']
            stage_to_torch = StageToCHWFloat(keys)
            staged_dataset = stage_to_torch(dataset)
            sample = dataset[0]
            staged_sample = staged_dataset[0]

            image = staged_sample['image']
            assert sample['image'].shape[2] == image.shape[0] == 3
            assert np.max(image) <= 1.
            assert np.min(image) >= 0.

            stage_to_numpy = StageToHWCUint8(keys)
            staged_dataset = stage_to_numpy(staged_dataset)
            staged_sample = staged_dataset[0]

            image = staged_sample['image']
            assert sample['image'].shape == image.shape
            assert np.max(image) <= 255
            assert np.min(image) >= 0

    def test_stage_cache(self, underfolder_folder):
        datasets = [
            UnderfolderDatabase(folder=underfolder_folder),
            UnderfolderDatabase(folder=underfolder_folder, use_lazy_samples=True)
        ]

        for dataset in datasets:
            stage_cache = StageCache()
            staged_dataset = stage_cache(dataset)
            sample = dataset[0]
            staged_sample = staged_dataset[0]

            assert sample.keys() == staged_sample.keys()
