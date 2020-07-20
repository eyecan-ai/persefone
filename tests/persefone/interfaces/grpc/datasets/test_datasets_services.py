
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

    @pytest.mark.mongo_real_server  # EXECUTE ONLY IF --mongo_real_server option is passed
    def test_lifecycle(self, temp_mongo_database, driver_temp_base_folder, minimnist_folder):
        self._test_lifecycle(temp_mongo_database, driver_temp_base_folder, minimnist_folder)

    @pytest.mark.mongo_mock_server
    def test_lifecycle_mock(self, temp_mongo_mock_database, driver_temp_base_folder, minimnist_folder):
        self._test_lifecycle(temp_mongo_mock_database, driver_temp_base_folder, minimnist_folder)

    def _test_lifecycle(self, mongo_client, driver_temp_base_folder, minimnist_folder):

        host = 'localhost'
        port = 10005

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
            assert client.new_dataset(dataset_name, category) is None, "Double creation should be invalid!"
            assert client.get_dataset(f'{dataset_name}IMPOSSIBLE_!!') is None, "Not found dataset should be not found!"

            assert dataset is not None, "Creation should be ok!"

            for sample_str, tree_items in tree.items():

                sample = client.new_sample(dataset_name, metadata={'sample': sample_str, 'items': [1, 2, 3]})
                assert sample is not None, "Sample creation should be ok!"
                assert client.new_sample(dataset_name + "_IMPOSSIBLE!!", metadata={}) is None, 'Impossible to create sample on dataset not found'
                print("SAMPLE", sample)
                assert 'metadata' in sample, "metadata key is missing"
                assert 'sample_id' in sample, "Sample ID is missing"

                retrieved_sample = client.get_sample(dataset_name, sample['sample_id'])
                assert retrieved_sample is not None, "Retrieved sample should be not None!"

                for item_name, filename in tree_items.items():

                    data, data_encoding = DataCoding.file_to_bytes(filename)
                    original_data = DataCoding.bytes_to_data(data, data_encoding)

                    item = client.new_item(dataset_name, sample['sample_id'], item_name, data, data_encoding)
                    assert client.new_item(dataset_name, sample['sample_id'], item_name, data, data_encoding) is None, (
                        "Item with same name not allowed!"
                    )
                    assert client.new_item(dataset_name + "XX!", sample['sample_id'], item_name, data, data_encoding) is None, (
                        "Should be wrong database"
                    )
                    assert client.new_item(dataset_name, len(tree_items.items()) + 100, item_name, data, data_encoding) is None, (
                        "Should be wrong sample id"
                    )
                    assert item is not None, "Item creation should be ok!"
                    assert 'name' in item, "name key is missing"

                    item_full = client.get_item(dataset_name, sample['sample_id'], item_name, fetch_data=True)
                    assert item_full is not None, "Item creation should be ok!"
                    assert client.get_item(dataset_name + 'XX!', sample['sample_id'], item_name, fetch_data=True) is None, (
                        "should be Wrong dataset!"
                    )
                    assert client.get_item(dataset_name, len(tree_items.items()) + 100, item_name, fetch_data=True) is None, (
                        "should be Wrong sample id!"
                    )
                    assert client.get_item(dataset_name, sample['sample_id'], item_name + "XX!", fetch_data=True) is None, (
                        "should be Wrong item name!"
                    )
                    assert 'name' in item_full, "name key is missing"
                    assert 'data' in item_full, "data key is missing"
                    assert len(item_full) > 0, "data base64 bytes should be not empty!"

                    item_data, item_data_encoding = client.get_item_data(dataset_name, sample['sample_id'], item_name)
                    assert len(item_data) > 0, "data bytes should be not empty!"

                    retrieved_data = DataCoding.bytes_to_data(item_data, item_data_encoding)
                    assert retrieved_data is not None, "Retrieved data should be not None"
                    assert original_data.shape == retrieved_data.shape, "Retrieved data shape is not valid"
                    assert np.array_equal(original_data, retrieved_data), "Retrieved data content is not valid"

                    fake_image = np.random.uniform(0, 255, (32, 32, 3)).astype(np.uint8)
                    fake_encoding = 'png'
                    fake_data = DataCoding.numpy_image_to_bytes(fake_image, fake_encoding)
                    item_updated = client.update_item(dataset_name, sample['sample_id'], item_name, fake_data, fake_encoding)
                    assert client.update_item(dataset_name + 'XX1', sample['sample_id'], item_name, fake_data, fake_encoding) is None, (
                        "Should be wrong dataset"
                    )
                    assert client.update_item(dataset_name, len(tree_items.items()) + 100, item_name, fake_data, fake_encoding) is None, (
                        "Should be wrong sample id"
                    )
                    assert client.update_item(dataset_name, sample['sample_id'], item_name + 'XX!', fake_data, fake_encoding) is None, (
                        "Should be wrong item name"
                    )
                    assert item_updated is not None, "Item update should be ok!"

                    item_data_2, item_data_encoding_2 = client.get_item_data(dataset_name, sample['sample_id'], item_name)
                    assert len(item_data_2) > 0, "data bytes should be not empty!"
                    retrieved_data_2 = DataCoding.bytes_to_data(item_data_2, item_data_encoding_2)
                    assert retrieved_data_2 is not None, "Retrieved data After Update should be not None"
                    assert fake_image.shape == retrieved_data_2.shape, "After UpdateRetrieved data shape is not valid"
                    assert np.array_equal(fake_image, retrieved_data_2), " After Update Retrieved data content is not valid"

                sample = client.update_sample(dataset_name, sample['sample_id'], metadata={'sample': 'update', 'items': [11]})
                print("U SAMPLE", sample)
                assert sample is not None, "Sample update should be ok!"
                assert client.update_sample(dataset_name + 'IMPOSSIBLE!', sample['sample_id'], metadata={}) is None, "Wrong update!"
                assert client.update_sample(dataset_name, len(tree_items.items()) + 100, metadata={}) is None, "Wrong update!"
                assert 'metadata' in sample, "metadata key is missing"
                assert sample['metadata']['sample'] == 'update', "After update 'sample' key should be 'update'"
                assert len(sample['metadata']['items']) == 1, "After update 'items' key should be one-sized list"

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
                    assert 'has_data' in item, "has_data key is missing"
                    assert item['has_data'] is False, "has_data should be false"

        for dataset_idx, dataset_name in enumerate(dataset_names):
            dataset = client.get_dataset(dataset_name, fetch_data=True)
            samples = dataset['samples']
            for sample in samples:
                for item in sample['items']:
                    assert len(item['data']) > 0, "item data should be not empty if  fetched"
                    assert item['has_data'] is True, "has_data should be true"

        assert len(client.datasets_list()) == len(dataset_names), "Datasets list is wrong!"

        for dataset_idx, dataset_name in enumerate(dataset_names):
            assert client.delete_dataset(dataset_name) is True, "Deletion should be ok!"
            assert client.delete_dataset(dataset_name) is False, "Double Deletion should be invalid!"

        assert len(client.datasets_list()) == 0, "Datasets list must be empty after armageddon!"

        print("TEMP FOLDER", driver_temp_base_folder)

        p = Path(driver_temp_base_folder)
        subitems = [x for x in p.glob('**/*') if x.is_file()]
        assert len(subitems) == 0, "No breadcrumbs please!"
        # for i in p.glob('**/*'):
        #     print(i.name)
