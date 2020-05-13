import pytest
from click.testing import CliRunner
from persefone.examples.tools.h5database_from_folder import h5database_from_folder


@pytest.fixture(scope="function")
def temp_dataset_file(tmpdir_factory):
    fn = tmpdir_factory.mktemp("data").join("_h5dataset_temp.h5")
    return fn


def test_cli(minimnist_folder, temp_dataset_file):

    opts = []
    opts.extend(['--folder', minimnist_folder])
    opts.extend(['--output_file', temp_dataset_file])

    runner = CliRunner()
    result = runner.invoke(h5database_from_folder, opts)
    assert result.exit_code == 0, "First creation should work!"
    result = runner.invoke(h5database_from_folder, opts)
    assert result.exit_code == 1, "Second creation should not work!"
    assert result.exception is not None, "Second creation should raise Exception!"
