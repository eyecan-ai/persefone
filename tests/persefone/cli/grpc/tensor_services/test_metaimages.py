from persefone.cli.grpc.tensor_services.metaimages_sample_client import metaimages_client
from persefone.cli.grpc.tensor_services.metaimages_sample_server import metaimages_server
from click.testing import CliRunner
from pathlib import Path


def test_metaimages_client(minimnist_folder):

    sample_image = str(list(Path(minimnist_folder).glob('*.jpg'))[0])

    runner = CliRunner()
    result = runner.invoke(metaimages_client, ['--target_image', sample_image, '--debug'])
    print(result.output)
    assert result.exit_code == 0
    assert 'Failed' in result.output


def test_metaimages_server():
    runner = CliRunner()
    result = runner.invoke(metaimages_server, ['--port', '100'])
    assert 'GRPC Port is not valid!' in result.output
