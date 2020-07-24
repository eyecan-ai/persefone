from persefone.interfaces.proto.utils.comm import ResponseStatusUtils
from persefone.data.io.drivers.common import AbstractFileDriver
from persefone.data.databases.mongo.clients import MongoDatabaseClient
from persefone.interfaces.grpc.deep_services_pb2_grpc import DeepServiceServicer, add_DeepServiceServicer_to_server
from persefone.interfaces.grpc.deep_services_pb2 import (
    DDeepServiceRequest, DDeepServiceResponse
)
import grpc
from abc import ABC, abstractmethod


class DeepService(ABC, DeepServiceServicer):

    def __init__(self):
        pass

    def register(self, grpc_server):
        add_DeepServiceServicer_to_server(self, grpc_server)

    @abstractmethod
    def DeepServe(self, request: DDeepServiceRequest, context: grpc.ServicerContext) -> DDeepServiceResponse:
        pass


class EchoDeepService(ABC, DeepServiceServicer):

    def __init__(self):
        pass

    def register(self, grpc_server):
        add_DeepServiceServicer_to_server(self, grpc_server)

    def DeepServe(self, request: DDeepServiceRequest, context: grpc.ServicerContext) -> DDeepServiceResponse:
        response = DDeepServiceResponse()
        response.status.CopyFrom(ResponseStatusUtils.create_ok_status("ok!"))
        response.metadata.CopyFrom(request.metadata)
        response.bundle.CopyFrom(request.bundle)
        return response


class MongoDeepService(DeepService):

    def __init__(self,
                 mongo_client: MongoDatabaseClient,
                 driver: AbstractFileDriver,
                 ):
        super(MongoDeepService, self).__init__()

        self._mongo_client = mongo_client
        assert isinstance(driver, AbstractFileDriver), f"Invalid driver: {driver}"
        self._drivers = [driver]

    def DeepServe(self, request, context):
        print("REUQEST", request)
        return DDeepServiceResponse()
