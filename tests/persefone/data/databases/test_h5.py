from persefone.data.databases.h5 import H5Database, H5DatabaseIO
from persefone.utils.filesystem import tree_from_underscore_notation_files
import pytest
import numpy as np
import uuid
from pathlib import Path


class TestH5Database(object):

    @pytest.fixture(scope="function")
    def temp_dataset_file(self, tmpdir_factory):
        fn = tmpdir_factory.mktemp("data").join("_h5dataset_temp.h5")
        return fn

    @pytest.fixture(scope="session")
    def sample_keys(self):
        key = str(uuid.uuid1())
        return [
            key,
            f'one/{key}',
            f'/3/2/1/{key}'
        ]

    @pytest.fixture(scope="session")
    def sample_items(self):
        return [
            {
                "shape": (512, 512, 3),
                "maxshape": None,
                "dtype": np.uint8,
                "name": 'map_rgb'
            },
            {
                "shape": (1, 512, 512),
                "maxshape": (None, 512, 512),
                "shape_random": (5, 512, 512),
                "dtype": np.float32,
                "name": 'pytorch_tensor'
            },
            {
                "shape": (1, 11, 1),
                "maxshape": (None, 11, None),
                "shape_random": (5, 11, 2),
                "dtype": np.float64,
                "name": 'double_none'
            },
            {
                "shape": (1, 2, 3, 4, 5, 6),
                "maxshape": None,
                "dtype": np.int32,
                "name": 'progressive_int32'
            }
        ]

    @classmethod
    def _generate_item(cls, index):
        return None

    def test_h5_database_keys(self, temp_dataset_file, sample_keys):
        print(temp_dataset_file)

        database = H5Database(filename=temp_dataset_file)
        with database:
            for key in sample_keys:
                key = H5Database.purge_key(key)
                print(f"Testing Key:{key}, {H5Database.purge_key(key)}")
                group = database.get_group(key, force_create=False)
                assert group is None, f"Group '{key}' must be None"
                group = database.get_group(key, force_create=True)
                assert group is not None, f"Group '{key}' must be not None"
                assert group.name == key, f"Group name is different from '{key}'"

    def test_h5_database_data(self, temp_dataset_file, sample_keys, sample_items):
        print(temp_dataset_file)

        database = H5Database(filename=temp_dataset_file)
        with database:
            for key in sample_keys:
                key = H5Database.purge_key(key)
                print(f"Testing Key:{key}, {H5Database.purge_key(key)}")
                group = database.get_group(key, force_create=True)
                assert group is not None, f"Group '{key}' must be not None"

                for item in sample_items:

                    data_name = item['name']
                    shape = item['shape']
                    maxshape = item['maxshape']
                    dtype = item['dtype']

                    print(f"Testing Data:{data_name}")
                    data = database.get_data(key, data_name)
                    assert data is None, f"Item Data '{data_name}' must be None"

                    database.create_data(key, data_name, shape=shape, maxshape=maxshape, dtype=dtype)
                    data = database.get_data(key, data_name)
                    assert data is not None, f"Item Data '{data_name}' must be not None"
                    assert data.shape == shape, f"Item Data '{data_name}' shape must be {shape}"

                    # Fill / Fetch
                    random_data = np.random.randint(0, 255, shape).astype(dtype)
                    data[...] = random_data
                    data_retrieve = database.get_data(key, data_name)
                    assert data[...].shape == shape, f"Item Data '{data_name}' shape must be {shape}"
                    assert np.array_equal(random_data, data_retrieve[...]), "Retrieved data is different from original"

                    if 'shape_random' in item:
                        shape_random = item['shape_random']
                        random_data = np.random.randint(0, 255, shape_random).astype(dtype)
                        data.resize(shape_random)
                        data[...] = random_data
                        data_retrieve = database.get_data(key, data_name)
                        assert data[...].shape == shape_random, f"Item Data '{data_name}' shape must be {shape}"
                        assert np.array_equal(random_data, data_retrieve[...]), "Retrieved data is different from original"


class TestH5DatabaseIO(object):

    @pytest.fixture(scope="function")
    def temp_dataset_file(self, tmpdir_factory):
        fn = tmpdir_factory.mktemp("data").join("_h5dataset_temp.h5")
        return fn

    @pytest.fixture(scope="session")
    def compression_configs(self):
        return {
            'nocompression': {
                'compression': None,
                'compression_opts': None
            },
            'gzip4': {
                'compression': 'gzip',
                'compression_opts': 4
            },
            'gzip1': {
                'compression': 'gzip',
                'compression_opts': 1
            },
            'gzip9': {
                'compression': 'gzip',
                'compression_opts': 9
            },
            'lzf': {
                'compression': 'lzf',
                'compression_opts': None
            }
        }

    def test_io_creation_from_undesrcore_notation_folder(self, temp_dataset_file, minimnist_folder):

        print(temp_dataset_file)

        database = H5DatabaseIO.generate_from_folder(
            h5file=temp_dataset_file,
            folder=minimnist_folder
        )

        tree = tree_from_underscore_notation_files(minimnist_folder)

        with database:

            for key, slots in tree.items():
                for item_name, filename in slots.items():
                    assert key in database.handle.keys(), f"Key {key} not present!"
                    loaded_data = H5DatabaseIO.load_data_from_file(filename)
                    if loaded_data is not None:
                        group = database.get_group(key)
                        assert item_name in group, f"Item with name {item_name} not present!"
                        data = database.get_data(key, item_name)
                        assert data is not None, f"Item with name {item_name} is None!"
                        assert data.shape == loaded_data.shape, f"Item with name {item_name} has wrong shape"

    def test_io_compressions(self, compression_configs, temp_dataset_file, minimnist_folder):

        database_configs = compression_configs
        for config_name, cfg in database_configs.items():
            database = H5DatabaseIO.generate_from_folder(
                h5file=temp_dataset_file,
                folder=minimnist_folder,
                **cfg
            )

            size = Path(database.filename).stat().st_size
            print(config_name, f"size: {size}")

            Path(temp_dataset_file).unlink()
