from pathlib import Path
from persefone.data.databases.filesystem.underfolder import (
    UnderfolderDatabase, UnderfolderDatabaseGenerator, UnderfolderLazySample
)
import numpy as np


class TestUnderscoreFolder(object):

    def test_lazysamples(self, underfolder_folder):

        print(underfolder_folder)
        dataset = UnderfolderDatabase(folder=underfolder_folder, use_lazy_samples=True)

        for sample_id in range(len(dataset)):
            sample = dataset[sample_id]
            assert isinstance(sample, UnderfolderLazySample)
            keys = list(sample.keys())
            for key in keys:
                assert isinstance(sample.get_path(key), Path)
                del sample[key]

    def test_creation(self, underfolder_folder):

        print(underfolder_folder)
        dataset = UnderfolderDatabase(folder=underfolder_folder)

        assert isinstance(dataset.metadata, dict)

        assert 'cfg' in dataset.metadata
        assert 'numbers' in dataset.metadata
        assert 'pose' in dataset.metadata

        assert len(dataset) == 20

        counter = 0
        for sample in dataset:

            assert 'image' in sample
            assert 'image_mask' in sample
            assert 'image_maskinv' in sample
            assert 'label' in sample
            assert 'metadata' in sample
            assert 'metadatay' in sample
            assert 'points' in sample

            assert isinstance(sample['image'], np.ndarray)
            assert isinstance(sample['image_mask'], np.ndarray)
            assert isinstance(sample['image_maskinv'], np.ndarray)
            counter += 1

        assert counter == len(dataset)

    def test_creation_plain_folder(self, underfolder_folder):

        underfolder_folder = Path(underfolder_folder) / UnderfolderDatabase.DATA_SUBFOLDER
        dataset = UnderfolderDatabase(folder=underfolder_folder)

        assert isinstance(dataset.metadata, dict)
        assert not dataset.metadata

        assert len(dataset) == 20

        counter = 0
        for sample in dataset:

            assert 'image' in sample
            assert 'image_mask' in sample
            assert 'image_maskinv' in sample
            assert 'label' in sample
            assert 'metadata' in sample
            assert 'metadatay' in sample
            assert 'points' in sample

            assert isinstance(sample['image'], np.ndarray)
            assert isinstance(sample['image_mask'], np.ndarray)
            assert isinstance(sample['image_maskinv'], np.ndarray)
            counter += 1

        assert counter == len(dataset)

    def test_skeleton(self, underfolder_folder):

        dataset = UnderfolderDatabase(folder=underfolder_folder)

        assert isinstance(dataset.metadata, dict)

        assert 'cfg' in dataset.metadata
        assert 'numbers' in dataset.metadata
        assert 'pose' in dataset.metadata

        assert len(dataset) == 20

        counter = 0
        for sample in dataset.skeleton:

            assert 'image' in sample
            assert 'image_mask' in sample
            assert 'image_maskinv' in sample
            assert 'label' in sample
            assert 'metadata' in sample
            assert 'metadatay' in sample
            assert 'points' in sample
            assert '_id' in sample

            assert isinstance(sample['image'], str)
            assert isinstance(sample['image_mask'], str)
            assert isinstance(sample['image_maskinv'], str)

            counter += 1

        assert counter == len(dataset)

    def test_remap(self, underfolder_folder):

        print(underfolder_folder)
        remap = {
            'image': 'x',
            'label': 'gt'
        }
        dataset = UnderfolderDatabase(folder=underfolder_folder, data_tags=remap)
        assert len(dataset) == 20

        counter = 0
        for sample in dataset:

            assert 'image' not in sample
            assert 'image_mask' not in sample
            assert 'image_maskinv' not in sample
            assert 'label' not in sample
            assert 'metadata' not in sample
            assert 'metadatay' not in sample
            assert 'points' not in sample
            assert 'x' in sample
            assert 'gt' in sample

            counter += 1

        assert counter == len(dataset)

    def test_creation_lazy_samples(self, underfolder_folder):

        print(underfolder_folder)
        dataset = UnderfolderDatabase(folder=underfolder_folder, use_lazy_samples=True)

        assert isinstance(dataset.metadata, dict)

        assert 'cfg' in dataset.metadata
        assert 'numbers' in dataset.metadata
        assert 'pose' in dataset.metadata

        assert len(dataset) == 20

        counter = 0
        for sample in dataset:

            assert 'image' in sample
            assert 'image_mask' in sample
            assert 'image_maskinv' in sample
            assert 'label' in sample
            assert 'metadata' in sample
            assert 'metadatay' in sample
            assert 'points' in sample

            assert isinstance(sample['image'], np.ndarray)
            assert isinstance(sample['image_mask'], np.ndarray)
            assert isinstance(sample['image_maskinv'], np.ndarray)
            counter += 1

        assert counter == len(dataset)

    def test_copy_database_metadata(self, underfolder_folder):
        prefix = '__'
        datasets = [
            UnderfolderDatabase(folder=underfolder_folder, use_lazy_samples=False, copy_database_metadata=prefix),
            UnderfolderDatabase(folder=underfolder_folder, use_lazy_samples=True, copy_database_metadata=prefix)
        ]

        for dataset in datasets:
            for sample in dataset:
                for k, v in dataset.metadata.items():
                    assert f'{prefix}{k}' in sample.keys()


class TestUnderscoreFolderCreation(object):

    def test_creation(self, underfolder_folder, generic_temp_folder):
        assert generic_temp_folder is not None

        print(generic_temp_folder)

        generator = UnderfolderDatabaseGenerator(generic_temp_folder)

        dataset = UnderfolderDatabase(folder=underfolder_folder)
        assert len(dataset) == 20

        for sample_id in range(len(dataset)):
            for k in dataset[sample_id].keys():
                filename = Path(dataset.skeleton[sample_id][k])
                extension = filename.suffix

                data = dataset[sample_id][k]
                generated_filename = generator.store_sample(sample_id, k, data, extension)
                print(sample_id, k, extension)
                print(generated_filename)

        generator.store_sample(-1, 'metadata', dataset.metadata, 'yml')

        reloaded_dataset = UnderfolderDatabase(folder=generic_temp_folder)

        assert len(reloaded_dataset) == len(dataset)

        for idx in range(len(dataset)):
            sample = dataset[idx]
            r_sample = reloaded_dataset[idx]
            assert sample.keys() == r_sample.keys()

    def test_export(self, underfolder_folder, generic_temp_folder):
        assert generic_temp_folder is not None

        print(generic_temp_folder)

        generator = UnderfolderDatabaseGenerator(generic_temp_folder)

        dataset = UnderfolderDatabase(folder=underfolder_folder)
        assert len(dataset) == 20

        generator.export_skeleton_database(dataset, generic_temp_folder)

        reloaded_dataset = UnderfolderDatabase(folder=generic_temp_folder)

        assert len(reloaded_dataset) == len(dataset)

        for idx in range(len(dataset)):
            sample = dataset[idx]
            r_sample = reloaded_dataset[idx]
            assert sample.keys() == r_sample.keys()
