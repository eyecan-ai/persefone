from persefone.utils.pyutils import get_arg
from persefone.data.databases.h5 import H5DatabaseIO
from persefone.data.databases.snapshot import SnapshotConfiguration, DatabaseSnapshot
from persefone.utils.configurations import XConfiguration
import pytest
from pathlib import Path
import schema
import hashlib
import numpy as np

SNAPSHOT_TEST_ROOT_KEYS = ['_items', 'miao', '/']
SNAPSHOT_TEST_CONFIGURATIONS = [
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
        'keys_equality': True
    }),

    # TEST 3
    ({
        'name': 'sample_snapshot',
        'sources': [],
    }, {
        'valid': True,
        'expected_size': 80,
        'keys_equality': True
    }),

    # TEST 4
    ({
        'name': 'sample_snapshot',
        'random_seed': 1114,
        'sources': [],
        'operations':[
            {'shuffle': 1},
        ]
    }, {
        'valid': True,
        'expected_size': 80,
        'keys_equality': False
    }),

    # TEST 5
    ({
        'name': 'sample_snapshot',
        'random_seed': 1.1,
        'sources': [],
    }, {
        'valid': False,
        'raises': schema.SchemaError
    }),
    # TEST 6
    ({
        'name': 'sample_snapshot',
        'random_seedx': 1,
        'sources': [],
    }, {
        'valid': False,
        'raises': schema.SchemaWrongKeyError
    })

]


SNAPSHOT_WITH_READERS_TEST_CONFIGURATIONS = [
    #  TEST FULL ARGUMENTS
    ({
        'name': 'sample_snapshot',
        'sources': [],
        'splits': {'train': 0.5, 'test': 0.5},
        'readers': {
            'columns': [
                'counter',
                'oddity',
                '@image',
                '@image_mask',
                '@image_maskinv',
                '@label',
                '@points'
            ]
        }
    }, {
        'valid': True
    }),

    ({
        'name': 'sample_snapshot',
        'sources': [],
        'splits': {'train': 0.5, 'test': 0.5},
        'readers': {
            'columns': [
                'counter',
                'oddity',
                'image',
                'points'
            ]
        }
    }, {
        'valid': True
    }),
    #
    ({
        'name': 'sample_snapshot',
        'sources': [],
        'splits': {'train': 0.5, 'test': 0.5},
        'readers': {
            'columns': [
                'image_mask',
                'image_maskinv',
            ]
        }
    }, {
        'valid': True
    }),

    # SAMPLE WITH READERS COLUMNS AS DICT
    ({
        'name': 'sample_snapshot',
        'sources': [],
        'splits': {'train': 0.5, 'test': 0.5},
        'readers': {
            'columns': {
                'image_mask': 'mask',
                'image_maskinv': 'mask_inv',
            }
        }
    }, {
        'valid': True
    }),

    # INVALIDS
    ({
        'name': 'sample_snapshot',
        'sources': [],
        'splits': {'train': 0.5, 'test': 0.5},
        'readersx': {
        }
    }, {
        'valid': False,
        'raises': schema.SchemaError
    }),
    #
    ({
        'name': 'sample_snapshot',
        'sources': [],
        'splits': {'train': 0.5, 'test': 0.5},
        'readers': {'column': []}
    }, {
        'valid': False,
        'raises': schema.SchemaError
    }),

]

SNAPSHOT_WITH_READERS_TEST_CONFIGURATIONS = [
    # SAMPLE WITH READERS COLUMNS AS DICT
    ({
        'name': 'sample_snapshot',
        'sources': [],
        'splits': {'train': 0.5, 'test': 0.5},
        'readers': {
            'columns': {
                'image_mask': 'mask',
                'image_maskinv': 'mask_inv',
            }
        }
    }, {
        'valid': True
    }),
]


class TestDatabaseSnapshot(object):

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

    @pytest.mark.parametrize("root_item", SNAPSHOT_TEST_ROOT_KEYS)
    @pytest.mark.parametrize("cfg, expectations", SNAPSHOT_TEST_CONFIGURATIONS)
    def test_snapshots_creation(self, root_item, cfg, expectations, temp_dataset_files_bunch, minimnist_folder, temp_yaml_file):

        for source in temp_dataset_files_bunch:
            H5DatabaseIO.generate_from_folder(
                h5file=source,
                folder=minimnist_folder,
                root_item=root_item,
                uuid_keys=True,
                root_metadata={'root_item': root_item}
            )

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
        snapshot = DatabaseSnapshot(cfg=snapshot_cfg)
        assert snapshot.is_valid(), "Snapshot configuration should be valid"

        # Len sources
        assert len(snapshot.sources) == len(cfg['sources']), "Sources size is wrong!"

        # print(snapshot.database.size)
        # print(snapshot.database.data.index)
        # print(snapshot.reduced_database.size)
        # print(snapshot.reduced_database.data.index)

        assert snapshot.reduced_database.size == expectations['expected_size'], "Expected size is wrong!"

        # print("L1\n", list(snapshot.database.data.index))
        # print("L2\n", list(snapshot.reduced_database.data.index))
        keys_before = list(snapshot.database.data.index)
        keys_after = list(snapshot.reduced_database.data.index)

        if expectations['keys_equality']:
            assert keys_before == keys_after, "Indices must be equals!"
        else:
            assert keys_before != keys_after, "Indices must be different!"

        # Splits
        output = snapshot.output
        if 'splits' in cfg:
            assert output.keys() == cfg['splits'].keys(), "Splits keys are wrong!"

            sumup = 0
            for db in output.values():
                sumup += db.size
            assert sumup == expectations['expected_size'], "Cumulative size is wrong!"
        else:
            assert len(output) == 1, "Without splits we don't want splits!"
            (output_name, output_db), = output.items()
            assert output_name == cfg['name'], "Snapshot name is wrong!"
            assert output_name == snapshot.name, "Snapshot name is wrong!"

    @pytest.mark.parametrize("root_item", SNAPSHOT_TEST_ROOT_KEYS)
    @pytest.mark.parametrize("cfg, expectations", SNAPSHOT_WITH_READERS_TEST_CONFIGURATIONS)
    def test_readers(self, root_item, cfg, expectations, temp_dataset_files_bunch, minimnist_folder, temp_yaml_file):

        for source in temp_dataset_files_bunch:
            H5DatabaseIO.generate_from_folder(
                h5file=source,
                folder=minimnist_folder,
                root_item=root_item,
                uuid_keys=True,
                root_metadata={'root_item': root_item}
            )

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
        snapshot = DatabaseSnapshot(cfg=snapshot_cfg)
        assert snapshot.is_valid(), "Snapshot configuration should be valid"

        # Readers
        readers = snapshot.output_readers
        assert len(readers) == len(snapshot.output), "Readers number is wrong!"

        for name, reader in readers.items():
            assert name in snapshot.output, f"Reader name '{name}' invalid!"
            assert len(reader) == len(snapshot.output[name]), f"Output dataset '{name}' wrong size!"
            assert reader.database == snapshot.output[name], "Databases mismatch!"

            if len(reader) > 0:
                item = reader[0]
                assert len(item.keys()) == len(cfg['readers']['columns']), "Columns in reader are wrong!"

    def test_readers_repeatability(self, temp_dataset_files_bunch, minimnist_folder, temp_yaml_file):

        cfg = {
            'name': 'sample_snapshot',
            'sources': [],
            'operations': [
                {'limit': 1000},
                {'shuffle': 1},
            ],
            'splits': {
                'train': 0.7,
                'test': 0.1,
                'val': 0.2
            },
            'readers': {'columns': {'image': 'grayscale', '_idx': 'id'}}
        }

        seeds = [10000, 1000, 10, 666, 4, 8, 15, 16, 23, 42]

        for source in temp_dataset_files_bunch:
            H5DatabaseIO.generate_from_folder(
                h5file=source,
                folder=minimnist_folder,
                root_item='_items',
                uuid_keys=True,
                root_metadata={'root_item': '_items'}
            )
            print("H5 File", source)

        images_sums = []
        hashes = []
        for seed in seeds:
            cfg['random_seed'] = seed

            # Set corresponding h5 files in the configuration
            cfg['sources'] = [str(x) for x in temp_dataset_files_bunch]

            # save config
            XConfiguration.from_dict(cfg).save_to(temp_yaml_file)
            print("YAML CFG", temp_yaml_file)

            snapshot_cfg = SnapshotConfiguration(filename=temp_yaml_file)
            assert snapshot_cfg.is_valid(), "Must be a valid configuration!"
            # Snapshot
            snapshot = DatabaseSnapshot(cfg=snapshot_cfg)
            print(snapshot)

            images_sum = 0.0
            indices = []
            for _, reader in snapshot.output_readers.items():
                for item in reader:
                    images_sum += item['grayscale'].sum()
                    indices.append(item['id'])

            images_sums.append(images_sum)

            indices = hashlib.md5(''.join(indices).encode('utf-8')).hexdigest()
            hashes.append(indices)

        images_sums = np.array(images_sums).astype(int)
        print("Sums:", images_sums)
        assert images_sums.max() == images_sums.min(), "Images pixels sum must be equal despite shuffle"

        hashes = np.array(hashes)

        assert len(hashes) == len(np.unique(hashes)), "The probability to have two ID order after shuffle is very low!"
