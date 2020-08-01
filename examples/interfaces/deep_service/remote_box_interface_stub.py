from concurrent import futures
from persefone.interfaces.grpc.servers.deep_services import EndpointDeepService
from persefone.interfaces.grpc.clients.deep_services import DeepServiceCFG, DeepServicePack
import grpc
from schema import Optional, Or, Schema
import json


def to_dict(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.__dict__))


class EBoxDataset(object):
    def __init__(self, name):
        self.name = name


class EBoxModel(object):
    def __init__(self):
        self.name = 'model'
        self.datasets = [
            EBoxDataset(name='D0'),
            EBoxDataset(name='D1')
        ]


# PING
class RemoteBoxInterfaceStub(object):

    def __init__(self):
        pass

    def ping(self, pack: DeepServicePack) -> DeepServicePack:
        reply_pack = DeepServicePack()
        return reply_pack

    def get_datasets(self, pack: DeepServicePack) -> DeepServicePack:
        reply_pack = DeepServicePack()
        reply_pack.metadata = {'datasets': []}
        for i in range(10):
            reply_pack.metadata['datasets'].append(to_dict(EBoxDataset(name=f'Data_{i}')))

        print("Reply", reply_pack.metadata)
        return reply_pack


# Create Callbackable service with Schemas map
service = EndpointDeepService()
stub = RemoteBoxInterfaceStub()

service.register_endpoint(
    name="ping",
    schema=Schema({
        'endpoint': '/ping',
        'method': 'GET'
    }),
    callback=stub.ping
)


# GET DATASETS
service.register_endpoint(
    name="get_datasets",
    schema=Schema({
        'endpoint': Or('/datasets', '/datasets/'),
        'method': 'GET',
        Optional('params'): {
            'name': str
        }
    }),
    callback=stub.get_datasets,
    output_schema=Schema({
        'datasets': [{'name': str}]
    })
)


# Create server
host = 'localhost'
port = 50051
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=DeepServiceCFG().options)
server.add_insecure_port(f'{host}:{port}')
service.register(server)
server.start()
server.wait_for_termination()
