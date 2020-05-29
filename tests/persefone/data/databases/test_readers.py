from persefone.utils.pyutils import get_arg
from persefone.data.databases.h5 import H5DatabaseIO, H5SimpleDatabase
from persefone.data.databases.readers import H5SimpleDataReader
from persefone.data.databases.snapshot import SnapshotConfiguration, DatabaseSnapshot
from persefone.utils.configurations import XConfiguration
import pytest
from pathlib import Path
import numpy as np

READERS_TEST_ROOT_KEYS = ['_items']
READERS_TEST_ENCODINGS_TAGS = [
    {},
    {'image': 'jpg', 'image_mask': 'png', 'image_maskinv': 'tiff'},
    {'image': 'bmp', 'image_mask': 'jpg', 'image_maskinv': 'jpg'},
]
READERS_TEST_CONFIGURATIONS = [
    #  TEST FULL ARGUMENTS
    ({
        'name': 'sample_snapshot',
        'random_seed': 666,
        'sources': [],
        'queries': [
                'counter >= 12',
                'oddity == 0'
        ],
        'operations': [
            {'limit': 80},
            {'shuffle': 1}
        ],
        'splits': {
            'train': 0.7,
            'test': 0.1,
            'val': 0.2
        }
    }, {
        'valid': True,
        'expected_size': 16,
        'images_shapes': {'image': (28, 28, 3)},
        'keys_equality': False
    }),

    # TEST 2
    ({
        'name': 'sample_snapshot',
        'sources': [],
        'queries': [
                'counter >= 0',
                'oddity != 10'
        ],
        'operations': [
        ],
        'splits': {
            'train': 0.7,
            'test': 0.1,
            'val': 0.2
        }
    }, {
        'valid': True,
        'expected_size': 80,
        'images_shapes': {'image': (28, 28, 3)},
        'keys_equality': True
    }),
]


class TestH5SimpleDataReader(object):

    @pytest.fixture(scope="function")
    def temp_dataset_files_bunch(self, tmpdir_factory):
        return [
            tmpdir_factory.mktemp("data").join("_h5dataset_temp_0.h5"),
            tmpdir_factory.mktemp("data").join("_h5dataset_temp_1.h5"),
            tmpdir_factory.mktemp("data").join("_h5dataset_temp_2.h5"),
            tmpdir_factory.mktemp("data").join("_h5dataset_temp_3.h5")
        ]

    @pytest.fixture(scope="function")
    def temp_yaml_file(self, tmpdir_factory):
        fn = tmpdir_factory.mktemp("data").join("configuration.yml")
        return fn

    @pytest.mark.parametrize("root_item", READERS_TEST_ROOT_KEYS)
    @pytest.mark.parametrize("encodings_tags", READERS_TEST_ENCODINGS_TAGS)
    @pytest.mark.parametrize("cfg, expectations", READERS_TEST_CONFIGURATIONS)
    def test_h5_simple_data_reader(self, root_item, encodings_tags, cfg, expectations, temp_dataset_files_bunch, minimnist_folder, temp_yaml_file):

        for source in temp_dataset_files_bunch:
            H5DatabaseIO.generate_from_folder(
                h5file=source,
                folder=minimnist_folder,
                root_item=root_item,
                uuid_keys=True,
                root_metadata={'root_item': root_item},
                image_compression_tags=encodings_tags
            )
            print(encodings_tags, "H5 File", source)

        # Set corresponding h5 files in the configuration
        cfg['sources'] = [str(x) for x in temp_dataset_files_bunch]

        # save config
        XConfiguration.from_dict(cfg).save_to(temp_yaml_file)
        print("YAML CFG", temp_yaml_file)

        snapshot_cfg = SnapshotConfiguration(filename=temp_yaml_file)

        if not expectations['valid']:
            raise_error = get_arg(expectations, 'raises', KeyError)
            with pytest.raises(raise_error):
                snapshot_cfg.validate()
            return

        assert snapshot_cfg.is_valid(), "Configuration cannot be invalid!"

        for source in snapshot_cfg.params.sources:
            assert Path(source).exists(), f"source {source} should exists!"

        # Snapshot
        snapshot = DatabaseSnapshot(filename=temp_yaml_file)

        for enable_cache in [False, True]:

            # ALL COLUMNS OF ```H5DatabaseIO.generate_from_folder``` generated database
            columns_groups = [
                ['counter', 'oddity', '@image', '@image_mask', '@image_maskinv', '@label', '@points'],
                ['counter', 'oddity', 'image', 'image_mask', 'image_maskinv', 'label', 'points'],

                {
                    'counter': 'counter2',
                    'oddity': 'oddity2',
                    'image': 'image2',
                    'image_mask': 'image_mask2',
                    'image_maskinv': 'image_maskinv2',
                    'label': 'label2',
                    'points': 'pointd2'
                },

            ]

            for columns in columns_groups:

                reader_sizes = 0
                for split_name, database in snapshot.output.items():

                    reader = H5SimpleDataReader(database, columns=columns, enable_cache=enable_cache)
                    reader_sizes += len(reader)

                    if isinstance(columns, list):
                        columns = dict(zip(columns, columns))

                    for item in reader:
                        for old_column, col in columns.items():
                            if col.startswith(H5SimpleDataReader.REFERENCE_PREFIX):
                                simple_col = col.replace(H5SimpleDataReader.REFERENCE_PREFIX, '', 1)
                                assert simple_col in item, f"key[REF] {simple_col} is missing!"
                                assert isinstance(item[simple_col], np.ndarray), f"Data is not a Numpy array, but {type(item[simple_col])}!"

                                images_shapes = expectations['images_shapes']

                                if simple_col in images_shapes:
                                    assert item[simple_col].shape == images_shapes[simple_col], f"Shape of '{simple_col}' is wrong!"
                                #print("SHAPE " * 10, simple_col, item[simple_col].shape)
                            else:
                                assert col in item, f"key {col} is missing!"

                    if enable_cache:
                        reader.close()

                assert reader_sizes == expectations['expected_size'], "Size of reader is wrong!"
