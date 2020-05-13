import pytest


@pytest.fixture()
def minimnist_folder():
    import pathlib
    return pathlib.Path(__file__).parent / 'sample_data/datasets/minimnist'
