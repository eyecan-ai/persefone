from persefone.data.databases.mongo.clients import MongoDatabaseClient, DatabaseDataset
from persefone.data.io.drivers.safefs import SafeFilesystemDriver, SafeFilesystemDriverCFG
import pytest
from pathlib import Path
from persefone.utils.filesystem import tree_from_underscore_notation_files


class TestDatabaseDataset(object):

    @pytest.fixture(scope='function')
    def mongo_client(self, mongo_configurations_folder):
        cfg_file = Path(mongo_configurations_folder) / 'mongo_test_client_cfg.yml'
        client = MongoDatabaseClient.create_from_configuration_file(filename=cfg_file)
        yield client
        #client.drop_database(key0=client.DROP_KEY_0, key1=client.DROP_KEY_1)
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
        return configurations_folder / 'drivers/securefs.yml'

    @pytest.fixture(scope="function")
    def driver_temp_base_folder(self, tmpdir_factory):
        fn = tmpdir_factory.mktemp("driver_folder")
        return fn

    def test_creation_mock(self,
                           mongo_client_mock,
                           safefs_sample_configuration,
                           minimnist_folder,
                           driver_temp_base_folder):

        self._test_creation(mongo_client_mock, safefs_sample_configuration, minimnist_folder, driver_temp_base_folder)

    @pytest.mark.mongo_real_server  # EXECUTE ONLY IF --mongo_real_server option is passed
    def test_creation(self,
                      mongo_client,
                      safefs_sample_configuration,
                      minimnist_folder,
                      driver_temp_base_folder):

        self._test_creation(mongo_client, safefs_sample_configuration, minimnist_folder, driver_temp_base_folder)

    def _test_creation(self,
                       mongo_client,
                       safefs_sample_configuration,
                       minimnist_folder,
                       driver_temp_base_folder):

        tree = tree_from_underscore_notation_files(minimnist_folder)
        # import pprint
        # pprint.pprint(tree)

        dataset_name = 'FAKEDATASET_TEMP'
        dataset_name = 'FAKEDATASET_CATEGORY_TEMP'

        print("Driver dataset folder: ", driver_temp_base_folder)

        cfg = SafeFilesystemDriverCFG.from_dict({'base_folder': driver_temp_base_folder})
        drivers = {
            SafeFilesystemDriver.driver_name(): SafeFilesystemDriver(cfg)
        }
        dataset = DatabaseDataset(mongo_client, dataset_name, dataset_name, drivers=drivers)

        for sample_str, items in tree.items():
            sample_idx = int(sample_str)
            sample = dataset.add_sample(sample_idx, {'sample': sample_idx * 2, 'items': items.keys()})
            assert sample is not None, "Sample should be not None!"
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

        dataset.delete(security_name=dataset_name)
