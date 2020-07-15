
from persefone.interfaces.grpc.servers.datasets_services import MongoDatasetService, DatasetsServiceCFG
from persefone.data.databases.mongo.clients import MongoDatabaseClient
from persefone.data.io.drivers.safefs import SafeFilesystemDriver
import grpc
from concurrent import futures
import logging
import click
from persefone.utils.cli_options import cli_base_options, cli_host_options


@click.command("Launch Datasets Services Server")
@cli_base_options
@cli_host_options
@click.option('--database_cfg', default='database.yml', help="Database configuration file", show_default=True)
@click.option('--driver_cfg', default='database_driver.yml', help="Database IO Driver configuration file", show_default=True)
def datasets_services_server(**options):

    logging.basicConfig(level=logging.DEBUG if options.get('debug') else logging.WARNING)

    driver = SafeFilesystemDriver.create_from_configuration_file(options.get('driver_cfg'))
    mongo_client = MongoDatabaseClient.create_from_configuration_file(filename=options.get('database_cfg'))

    cfg = DatasetsServiceCFG()
    service = MongoDatasetService(mongo_client=mongo_client, driver=driver)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=cfg.options)
    server.add_insecure_port(f'{options.get("host")}:{options.get("port")}')

    service.register(server)

    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    datasets_services_server()
