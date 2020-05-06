import pytest
import numpy as np
import uuid
from persefone.data.io.h5dataset import H5Dataset


@pytest.mark.data_io_h5dataset
class TestH5Dataset(object):

    @pytest.fixture(scope="session")
    def temp_dataset_file(self, tmpdir_factory):
        fn = tmpdir_factory.mktemp("data").join("_h5dataset_temp.h5")
        return fn

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

    def test_h5_dataset(self, temp_dataset_file, sample_items):

        print("TEMPO FILE", temp_dataset_file)
        dataset = H5Dataset(temp_dataset_file)

        with dataset:
            id = str(uuid.uuid1())
            item = dataset.get_item(id, force_create=False)
            assert item is None, f"Item '{id}' must be None"
            item = dataset.get_item(id, force_create=True)
            assert item is not None, f"Item '{id}' must be not None"

            for item in sample_items:

                data_name = item['name']
                shape = item['shape']
                maxshape = item['maxshape']
                dtype = item['dtype']
                data = dataset.get_data(id, data_name)

                assert data is None, f"Item Data '{data_name}' must be None"

                dataset.create_data(id, data_name, shape=shape, maxshape=maxshape, dtype=dtype)
                data = dataset.get_data(id, data_name)
                assert data is not None, f"Item Data '{data_name}' must be not None"
                assert data.shape == shape, f"Item Data '{data_name}' shape must be {shape}"

                # Fill / Fetch
                random_data = np.random.randint(0, 255, shape).astype(dtype)
                data[...] = random_data
                data_retrieve = dataset.get_data(id, data_name)
                assert data[...].shape == shape, f"Item Data '{data_name}' shape must be {shape}"
                assert np.array_equal(random_data, data_retrieve[...]), "Retrieved data is different from original"

                if 'shape_random' in item:
                    shape_random = item['shape_random']
                    random_data = np.random.randint(0, 255, shape_random).astype(dtype)
                    data.resize(shape_random)
                    data[...] = random_data
                    data_retrieve = dataset.get_data(id, data_name)
                    assert data[...].shape == shape_random, f"Item Data '{data_name}' shape must be {shape}"
                    assert np.array_equal(random_data, data_retrieve[...]), "Retrieved data is different from original"
