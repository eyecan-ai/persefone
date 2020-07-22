from persefone.data.databases.mongo.clients import MongoDatabaseClient, MongoDataset
from persefone.data.databases.mongo.readers import MongoDatasetReader
from persefone.interfaces.grpc.servers.datasets_services import MongoDatasetService, DatasetsServiceCFG
from persefone.utils.cli_options import cli_host_options
from persefone.data.io.drivers.safefs import SafeFilesystemDriver
import click
import cv2
import grpc
from concurrent import futures
import threading


@click.command("Converts H5 dataset into MongoDB Dataset")
@cli_host_options
@click.option('--database_cfg', default='database.yml', help="Database configuration file", show_default=True)
@click.option('--driver_cfg', default='database_driver.yml', help="Database IO Driver configuration file", show_default=True)
def mongo_datasets_viewer(host, port, database_cfg, driver_cfg):

    driver = SafeFilesystemDriver.create_from_configuration_file(driver_cfg)
    mongo_client = MongoDatabaseClient.create_from_configuration_file(filename=database_cfg)

    cfg = DatasetsServiceCFG()
    service = MongoDatasetService(mongo_client=mongo_client, driver=driver)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=cfg.options)
    server.add_insecure_port(f'{host}:{port}')

    service.register(server)

    def _serve():
        server.start()
        server.wait_for_termination()

    t = threading.Thread(target=_serve, daemon=True)
    print("Started on: ", f'{host}:{port}')
    t.start()
    t.join()

    # server.start()
    # server.wait_for_termination()


if __name__ == "__main__":
    mongo_datasets_viewer()
