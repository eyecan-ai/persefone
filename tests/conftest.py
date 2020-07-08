import pytest


@pytest.fixture()
def minimnist_folder():
    import pathlib
    return pathlib.Path(__file__).parent / 'sample_data/datasets/minimnist'


@pytest.fixture()
def configurations_folder():
    import pathlib
    return pathlib.Path(__file__).parent / 'sample_data/configurations/'


@pytest.fixture()
def augmentations_folder():
    import pathlib
    return pathlib.Path(__file__).parent / 'sample_data/augmentations'


def pytest_addoption(parser):
    parser.addoption("--mongo_real_server", action="store_true",
                     help="run the tests only in case of real localhost mongo server active")


def pytest_runtest_setup(item):
    if 'mongo_real_server' in item.keywords and not item.config.getoption("--mongo_real_server"):
        pytest.skip("need --mongo_real_server for full testing on real mongodb server")
