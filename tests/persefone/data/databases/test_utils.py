import pytest
from persefone.data.databases.pandas import PandasDatabaseIO
from persefone.data.databases.h5 import H5DatabaseIO, H5SimpleDatabase
from persefone.data.databases.utils import H5SimpleDatabaseUtils
from pathlib import Path
import uuid


class TestH5SimpleDatabaseUtils(object):

    @pytest.fixture(scope="function")
    def temp_dataset_files_bunch(self, tmpdir_factory):
        return [
            tmpdir_factory.mktemp("data").join("_h5dataset_temp_0.h5"),
            tmpdir_factory.mktemp("data").join("_h5dataset_temp_1.h5"),
            tmpdir_factory.mktemp("data").join("_h5dataset_temp_2.h5"),
            tmpdir_factory.mktemp("data").join("_h5dataset_temp_3.h5")
        ]

    @pytest.fixture(scope="function")
    def temp_tabular_file(self, tmpdir_factory):
        fn = tmpdir_factory.mktemp("data").join("_tabular.csv")
        return fn

    def test_h5files_to_pandas_database(self, temp_dataset_files_bunch, temp_tabular_file, minimnist_folder):

        for temp_dataset_file in temp_dataset_files_bunch:
            H5DatabaseIO.generate_from_folder(
                h5file=temp_dataset_file,
                folder=minimnist_folder,
                root_item='_items',
                uuid_keys=True
            )
            print(temp_dataset_file)

        cumulative_size = 0
        cumulative_keys = []
        for h5file in temp_dataset_files_bunch:
            simple_database = H5SimpleDatabase(filename=h5file)
            simple_database.open()
            cumulative_size += len(simple_database)
            cumulative_keys += simple_database.keys
            simple_database.close()

        pandas_database = H5SimpleDatabaseUtils.h5files_to_pandas_database(temp_dataset_files_bunch, include_filenames=True)
        PandasDatabaseIO.save_csv(pandas_database, temp_tabular_file)

        merged_database = PandasDatabaseIO.load_csv(temp_tabular_file)

        assert merged_database.size == cumulative_size, "Cumulative size is wrong!"

        for key in cumulative_keys:
            assert key in merged_database.data.index, f"Index {key} is missing!"

        test_columns = ['image', 'image_mask', 'image_maskinv', 'label']

        for column in test_columns:
            ref_column = f'{H5SimpleDatabase.DEFAULT_TABULAR_REFERENCE_TOKEN}{column}'
            assert ref_column in merged_database.data.columns, f"{ref_column} column is missing in tabular representation"

        for idx in range(merged_database.size):
            filename = merged_database.data.iloc[idx][f'{H5SimpleDatabase.DEFAULT_TABULAR_PRIVATE_TOKEN}filename']
            filename = Path(filename)
            assert filename.exists(), f"{filename} does not exist!"

            h5database = H5SimpleDatabase(filename=filename)

            with h5database:
                for column in test_columns:
                    ref_column = f'{H5SimpleDatabase.DEFAULT_TABULAR_REFERENCE_TOKEN}{column}'
                    ref_data_key = merged_database.data.iloc[idx][ref_column]
                    ref = h5database[ref_data_key]
                    assert ref is not None, f"column value ref to {ref_column} is None"

                    with pytest.raises(KeyError):
                        h5database[ref_data_key + "_" + str(uuid.uuid1())]
