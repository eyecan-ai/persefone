from persefone.data.databases.mongo.clients import MongoDatabaseClient, MongoDataset
from persefone.data.databases.mongo.repositories import ItemsRepository, DatasetsRepository, DatasetCategoryRepository, SamplesRepository
from persefone.data.io.drivers.safefs import SafeFilesystemDriver, SafeFilesystemDriverCFG
import pytest
from pathlib import Path
from persefone.utils.filesystem import tree_from_underscore_notation_files


class TestMongoDatabaseDataset(object):

    @pytest.mark.mongo_mock_server  # NOT EXECUTE IF --mongo_real_server option is passed
    def test_creation_mock(self,
                           temp_mongo_mock_database: MongoDatabaseClient,
                           safefs_sample_configuration,
                           minimnist_folder,
                           driver_temp_base_folder):

        self._test_creation(temp_mongo_mock_database, safefs_sample_configuration, minimnist_folder, driver_temp_base_folder)
        self._test_creation_recursive_delete(temp_mongo_mock_database,
                                             safefs_sample_configuration,
                                             minimnist_folder,
                                             driver_temp_base_folder
                                             )

    @pytest.mark.mongo_real_server  # EXECUTE ONLY IF --mongo_real_server option is passed
    def test_creation(self,
                      temp_mongo_database: MongoDatabaseClient,
                      safefs_sample_configuration,
                      minimnist_folder,
                      driver_temp_base_folder):

        self._test_creation(temp_mongo_database, safefs_sample_configuration, minimnist_folder, driver_temp_base_folder)
        self._test_creation_recursive_delete(temp_mongo_database, safefs_sample_configuration, minimnist_folder, driver_temp_base_folder)

    def _test_creation(self,
                       mongo_client,
                       safefs_sample_configuration,
                       minimnist_folder,
                       driver_temp_base_folder):

        tree = tree_from_underscore_notation_files(minimnist_folder)

        dataset_name = 'FAKEDATASET_TEMP'
        category_name = 'FAKEDATASET_CATEGORY_TEMP'

        print("Driver dataset folder: ", driver_temp_base_folder)

        cfg = SafeFilesystemDriverCFG.from_dict({'base_folder': driver_temp_base_folder})
        drivers = {
            SafeFilesystemDriver.driver_name(): SafeFilesystemDriver(cfg)
        }
        dataset = MongoDataset(mongo_client, dataset_name, category_name, drivers=drivers)

        for sample_str, items in tree.items():
            sample = dataset.add_sample({'sample': sample_str, 'items': items.keys()})
            assert sample is not None, "Sample should be not None!"
            sample_idx = sample.sample_id
            sample_r = dataset.get_sample(sample_idx)
            assert sample_r is not None, "Retrieved Sample should be not None!"
            assert sample == sample_r, "Retrieved sample is wrong!"

            for item_name, filename in items.items():
                item = dataset.add_item(sample_idx, item_name)
                assert item is not None, "Item should be not None!"
                item_r = dataset.get_item(sample_idx, item_name)
                assert item_r is not None, "Retrieved Item should be not None!"
                assert item == item_r, "Retrieved item is wrong!"

                resource = dataset.push_resource(
                    sample_idx,
                    item_name,
                    item_name,
                    filename,
                    SafeFilesystemDriver.driver_name()
                )
                assert resource is not None, "Failed to create resource"

                blob = dataset.fetch_resource_to_blob(resource)

                with open(filename, 'rb') as fin:
                    blob_gt = fin.read()
                assert blob == blob_gt, "Stored blob is strange!"

        dataset.delete(security_name=dataset_name)
        DatasetCategoryRepository.delete_category(name=category_name)
        assert DatasetsRepository.get_dataset(dataset_name=dataset_name) is None, "Dataset should be deleted!"
        assert SamplesRepository.count_samples() == 0, "No samples should be there!"

    def _test_creation_recursive_delete(self,
                                        mongo_client,
                                        safefs_sample_configuration,
                                        minimnist_folder,
                                        driver_temp_base_folder):

        tree = tree_from_underscore_notation_files(minimnist_folder)

        dataset_name = 'FAKEDATASET_TEMPRECURSIVE'
        category_name = 'FAKEDATASET_TEMPRECURSIVE_CATEGORY_TEMP'

        print("Driver BLOB dataset folder: ", driver_temp_base_folder)

        cfg = SafeFilesystemDriverCFG.from_dict({'base_folder': driver_temp_base_folder})
        drivers = {
            SafeFilesystemDriver.driver_name(): SafeFilesystemDriver(cfg)
        }
        dataset = MongoDataset(mongo_client, dataset_name, category_name, drivers=drivers)

        for sample_str, items in tree.items():

            sample = dataset.add_sample({'sample': sample_str, 'items': items.keys(), 'even': int(sample_str) % 2 == 0})
            assert sample is not None, "Sample should be not None!"
            sample_idx = sample.sample_id
            sample_r = dataset.get_sample(sample_idx)
            assert sample_r is not None, "Retrieved Sample should be not None!"
            assert sample == sample_r, "Retrieved sample is wrong!"

            for item_name, filename in items.items():
                item = dataset.add_item(sample_idx, item_name)
                assert item is not None, "Item should be not None!"
                item_r = dataset.get_item(sample_idx, item_name)
                assert item_r is not None, "Retrieved Item should be not None!"
                assert item == item_r, "Retrieved item is wrong!"

                blob = None
                with open(filename, 'rb') as fin:
                    blob = fin.read()

                resource = dataset.push_resource_from_blob(
                    sample_idx,
                    item_name,
                    item_name,
                    blob,
                    Path(filename).suffix,
                    SafeFilesystemDriver.driver_name()
                )
                assert resource is not None, "Failed to create resource"

                blob_retrieved = dataset.fetch_resource_to_blob(resource)
                assert blob == blob_retrieved, "Stored blob is strange!"

        samples = dataset.get_samples()
        for sample in samples:
            SamplesRepository.delete_sample(dataset_name=dataset_name, sample_idx=sample.sample_id)
        assert SamplesRepository.count_samples() == 0, "No samples should be there!"
        assert len(ItemsRepository.get_items()) == 0, "No items should be there"
        dataset.delete(security_name=dataset_name)
        DatasetCategoryRepository.delete_category(name=category_name)
        assert DatasetsRepository.get_dataset(dataset_name=dataset_name) is None, "Dataset should be deleted!"
