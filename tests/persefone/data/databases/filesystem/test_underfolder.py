from pathlib import Path
from persefone.data.databases.filesystem.underfolder import UnderfolderDatabase
import numpy as np


class TestUnderscoreFolder(object):

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
