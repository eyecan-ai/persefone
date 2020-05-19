from persefone.utils.pyutils import get_arg
from persefone.data.databases.h5 import H5DatabaseIO
from persefone.data.databases.snapshot import SnapshotConfiguration, DatabaseSnapshot
from persefone.utils.configurations import XConfiguration
import pytest
from pathlib import Path
import schema


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

    @pytest.mark.parametrize("cfg, expectations", SNAPSHOT_TEST_CONFIGURATIONS)
    def test_simple(self, cfg, expectations, temp_dataset_files_bunch, minimnist_folder, temp_yaml_file):

        for source in temp_dataset_files_bunch:
            H5DatabaseIO.generate_from_folder(
                h5file=source,
                folder=minimnist_folder,
                root_item='_items',
                uuid_keys=True
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
        snapshot = DatabaseSnapshot(filename=temp_yaml_file)
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
