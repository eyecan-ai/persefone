
import pytest
from persefone.data.databases.mongo.clients import MongoDatabaseClient
from persefone.data.io.drivers.safefs import SafeFilesystemDriver, SafeFilesystemDriverCFG
from pathlib import Path
from persefone.utils.filesystem import tree_from_underscore_notation_files
from persefone.interfaces.grpc.servers.datasets_services import MongoDatasetService, DatasetsServiceCFG
from persefone.interfaces.grpc.clients.datasets_services import DatasetsSimpleServiceClient
from persefone.utils.bytes import DataCoding
import grpc

from concurrent import futures
import threading
import numpy as np


class TestMongoDatasetService(object):

    @pytest.fixture(scope='function')
    def mongo_client(self, mongo_configurations_folder):
        cfg_file = Path(mongo_configurations_folder) / 'mongo_test_client_cfg.yml'
        client = MongoDatabaseClient.create_from_configuration_file(filename=cfg_file)
        yield client
        client.drop_database(key0=client.DROP_KEY_0, key1=client.DROP_KEY_1)
        client.disconnect()

    @pytest.fixture(scope='function')
    def mongo_client_mock(self, mongo_configurations_folder):
        cfg_file = Path(mongo_configurations_folder) / 'mongo_test_client_cfg_mock.yml'
        client = MongoDatabaseClient.create_from_configuration_file(filename=cfg_file)
        yield client
        client.drop_database(key0=client.DROP_KEY_0, key1=client.DROP_KEY_1)
        client.disconnect()

    @pytest.fixture
    def safefs_sample_configuration(self, configurations_folder):
        from pathlib import Path
        return Path(configurations_folder) / 'drivers/securefs.yml'

    @pytest.fixture(scope="function")
    def driver_temp_base_folder(self, tmpdir_factory):
        fn = tmpdir_factory.mktemp("driver_folder")
        return fn

    @pytest.mark.mongo_real_server  # EXECUTE ONLY IF --mongo_real_server option is passed
    def test_lifecycle(self, mongo_client, driver_temp_base_folder, minimnist_folder):
        self._test_lifecycle(mongo_client, driver_temp_base_folder, minimnist_folder)

    def test_lifecycle_mock(self, mongo_client_mock, driver_temp_base_folder, minimnist_folder):
        self._test_lifecycle(mongo_client_mock, driver_temp_base_folder, minimnist_folder)

    def _test_lifecycle(self, mongo_client, driver_temp_base_folder, minimnist_folder):

        host = 'localhost'
        port = 10005

        print("TEMP FOLDER", driver_temp_base_folder)
        cfg = SafeFilesystemDriverCFG.from_dict({'base_folder': driver_temp_base_folder})
        driver = SafeFilesystemDriver(cfg)

        cfg = DatasetsServiceCFG()
        service = MongoDatasetService(mongo_client=mongo_client, driver=driver)

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=cfg.options)
        server.add_insecure_port(f'{host}:{port}')

        service.register(server)

        def _serve():
            server.start()
            server.wait_for_termination()

        t = threading.Thread(target=_serve, daemon=True)
        t.start()

        tree = tree_from_underscore_notation_files(minimnist_folder)

        client = DatasetsSimpleServiceClient(host=host, port=port)

        dataset_names = [f'dataset_{x}' for x in range(4)]

        for dataset_idx, dataset_name in enumerate(dataset_names):
            category = f'category_{str(dataset_idx % 2)}'
            dataset = client.new_dataset(dataset_name, category)

            assert dataset is not None, "Creation should be ok!"

            for sample_str, tree_items in tree.items():

                sample = client.new_sample(dataset_name, metadata={'sample': sample_str, 'items': [1, 2, 3]})
                assert sample is not None, "Sample creation should be ok!"
                print("SAMPLE", sample)
                assert 'sample_id' in sample, "Sample ID is missing"

                for item_name, filename in tree_items.items():

                    data, data_encoding = DataCoding.file_to_bytes(filename)
                    original_data = DataCoding.bytes_to_data(data, data_encoding)

                    item = client.new_item(dataset_name, sample['sample_id'], item_name, data, data_encoding)
                    assert item is not None, "Item creation should be ok!"
                    assert 'name' in item, "name key is missing"

                    item_full = client.get_item(dataset_name, sample['sample_id'], item_name, fetch_data=True)
                    assert item_full is not None, "Item creation should be ok!"
                    assert 'name' in item_full, "name key is missing"
                    assert 'data' in item_full, "data key is missing"
                    assert len(item_full) > 0, "data base64 bytes should be not empty!"

                    item_data, item_data_encoding = client.get_item_data(dataset_name, sample['sample_id'], item_name)
                    assert len(item_data) > 0, "data bytes should be not empty!"

                    retrieved_data = DataCoding.bytes_to_data(item_data, item_data_encoding)
                    assert retrieved_data is not None, "Retrieved data should be not None"
                    assert original_data.shape == retrieved_data.shape, "Retrieved data shape is not valid"
                    assert np.array_equal(original_data, retrieved_data), "Retrieved data content is not valid"

        for dataset_idx, dataset_name in enumerate(dataset_names):
            dataset = client.get_dataset(dataset_name, fetch_data=False)
            assert dataset is not None, "Retrieved dataset should be not None"
            assert 'samples' in dataset, "sample key is missing"

            samples = dataset['samples']
            assert len(samples) > 0, "Samples number is wrong"

            for sample in samples:
                assert 'sample_id' in sample, "sample_id key is missing"
                assert 'items' in sample, "items key is missing"

                for item in sample['items']:
                    assert 'name' in item, "name key is missing"
                    assert len(item['data']) == 0, "item data should be empty if not fetched explicity"

        for dataset_idx, dataset_name in enumerate(dataset_names):
            dataset = client.get_dataset(dataset_name, fetch_data=True)
            samples = dataset['samples']
            for sample in samples:
                for item in sample['items']:
                    assert len(item['data']) > 0, "item data should be not empty if  fetched"

        assert len(client.datasets_list()) == len(dataset_names), "Datasets list is wrong!"
