
from persefone.interfaces.grpc.servers.datasets_services import MongoDatasetService, DatasetsServiceCFG
from persefone.data.databases.mongo.clients import MongoDatabaseClient
from persefone.data.io.drivers.safefs import SafeFilesystemDriver
import grpc
from concurrent import futures

driver = SafeFilesystemDriver.create_from_configuration_file('securefs_driver.yml')
mongo_client = MongoDatabaseClient.create_from_configuration_file(filename='database.yml')

cfg = DatasetsServiceCFG()
service = MongoDatasetService(mongo_client=mongo_client, driver=driver)


server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=cfg.options)
server.add_insecure_port(f'localhost:50051')

service.register(server)
service.register(server)
service.register(server)

server.start()
server.wait_for_termination()
