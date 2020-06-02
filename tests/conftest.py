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
