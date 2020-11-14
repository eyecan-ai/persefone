import pytest
from persefone.data.databases.mongo.clients import MongoDatabaseClient
from pathlib import Path


@pytest.fixture()
def minimnist_folder():
    import pathlib
    return pathlib.Path(__file__).parent / 'sample_data/datasets/minimnist'


@pytest.fixture()
def underfolder_folder():
    import pathlib
    return pathlib.Path(__file__).parent / 'sample_data/datasets/underfolder'


@pytest.fixture()
def underfoldertomix_folder():
    import pathlib
    return pathlib.Path(__file__).parent / 'sample_data/datasets/underfolder_tomix'


@pytest.fixture()
def configurations_folder():
    import pathlib
    return pathlib.Path(__file__).parent / 'sample_data/configurations/'


@pytest.fixture()
def augmentations_folder():
    import pathlib
    return pathlib.Path(__file__).parent / 'sample_data/augmentations'


@pytest.fixture(scope="function")
def generic_temp_folder(tmpdir_factory):
    fn = tmpdir_factory.mktemp("_generic_persefone_test_temp")
    return fn

# MONGO


def pytest_addoption(parser):
    parser.addoption("--mongo_real_server", action="store_true",
                     help="run the tests only in case of real localhost mongo server active")


def pytest_runtest_setup(item):
    if 'mongo_real_server' in item.keywords and not item.config.getoption("--mongo_real_server"):
        pytest.skip("need --mongo_real_server for full testing on real mongodb server")
    if 'mongo_mock_server' in item.keywords and item.config.getoption("--mongo_real_server"):
        pytest.skip("remove --mongo_real_server for testing on mock mongodb server")


@pytest.fixture
def safefs_sample_configuration(configurations_folder):
    from pathlib import Path
    return Path(configurations_folder) / 'drivers/securefs.yml'


@pytest.fixture()
def mongo_configurations_folder(configurations_folder):
    return configurations_folder / 'mongo'


@pytest.fixture(scope="function")
def driver_temp_base_folder(tmpdir_factory):
    fn = tmpdir_factory.mktemp("driver_folder")
    return fn


@pytest.fixture(scope='function')
def temp_mongo_database_keep_alive():
    cfg_file = Path(mongo_configurations_folder) / 'mongo_test_client_cfg.yml'
    client = MongoDatabaseClient.create_from_configuration_file(filename=cfg_file)
    yield client.connect()
    # client.drop_database()
    client.disconnect()


@pytest.fixture(scope='function')
def temp_mongo_database(mongo_configurations_folder):
    cfg_file = Path(mongo_configurations_folder) / 'mongo_test_client_cfg.yml'
    client = MongoDatabaseClient.create_from_configuration_file(filename=cfg_file)
    yield client.connect()
    client.drop_database(key0=client.DROP_KEY_0, key1=client.DROP_KEY_1)
    client.disconnect()


@pytest.fixture(scope='function')
def temp_mongo_persistent_database(mongo_configurations_folder):
    cfg_file = Path(mongo_configurations_folder) / 'mongo_test_persistent_client_cfg.yml'
    client = MongoDatabaseClient.create_from_configuration_file(filename=cfg_file)
    yield client.connect()
    # client.drop_database(key0=client.DROP_KEY_0, key1=client.DROP_KEY_1)
    client.disconnect()


@pytest.fixture(scope='function')
def temp_mongo_mock_database(mongo_configurations_folder):
    cfg_file = Path(mongo_configurations_folder) / 'mongo_test_client_cfg_mock.yml'
    client = MongoDatabaseClient.create_from_configuration_file(filename=cfg_file)
    yield client.connect()
    client.drop_database(key0=client.DROP_KEY_0, key1=client.DROP_KEY_1)
    client.disconnect()
