from persefone.interfaces.proto.utils.dtensor import DTensorUtils
from persefone.interfaces.proto.utils.comm import MetadataUtils, ResponseStatusUtils
from persefone.interfaces.grpc.deep_services_pb2_grpc import DeepServiceStub
from persefone.interfaces.grpc.deep_services_pb2 import (
    DDeepServiceRequest, DDeepServiceResponse
)
import grpc


class DeepServiceCFG(object):
    DEFAULT_MAX_MESSAGE_LENGTH = -1

    def __init__(self):
        self.options = [
            ('grpc.max_send_message_length', self.DEFAULT_MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', self.DEFAULT_MAX_MESSAGE_LENGTH),
        ]


class DeepServiceClient(object):

    def __init__(self, host='localhost', port=50051, cfg=DeepServiceCFG()):
        if isinstance(port, str):
            port = int(port)
        self._channel = grpc.insecure_channel(f'{host}:{port}', options=cfg.options)
        self._stub = DeepServiceStub(self._channel)

    def DeepServe(self, request: DDeepServiceRequest) -> DDeepServiceResponse:
        return self._stub.DeepServe(request)


class DeepServicePack:

    def __init__(self):
        self.metadata = {}
        self.arrays = []
        self.arrays_action = ''
        self.status_code = 0
        self.status_message = ''

    @property
    def valid(self):
        return self.status_code == ResponseStatusUtils.STATUS_CODE_OK

    def to_deep_service_request(self):
        request = DDeepServiceRequest()
        request.bundle.CopyFrom(DTensorUtils.numpy_to_dtensor_bundle(self.arrays, self.arrays_action))
        MetadataUtils.dict_to_struct(self.metadata, request.metadata)
        return request

    def to_request(self) -> DDeepServiceRequest:
        request = DDeepServiceRequest()
        request.bundle.CopyFrom(DTensorUtils.numpy_to_dtensor_bundle(self.arrays, self.arrays_action))
        MetadataUtils.dict_to_struct(self.metadata, request.metadata)
        return request

    def to_response(self) -> DDeepServiceResponse:
        response = DDeepServiceResponse()
        response.bundle.CopyFrom(DTensorUtils.numpy_to_dtensor_bundle(self.arrays, self.arrays_action))
        MetadataUtils.dict_to_struct(self.metadata, response.metadata)
        return response

    @classmethod
    def from_deep_service_response(cls, response: DDeepServiceResponse):
        pack = DeepServicePack()
        pack.metadata = MetadataUtils.struct_to_dict(response.metadata)
        pack.arrays, pack.arrays_action = DTensorUtils.dtensor_bundle_to_numpy(response.bundle)
        pack.status_code = response.status.code
        pack.status_message = response.status.message
        return pack

    @classmethod
    def from_deep_service_request(cls, request: DDeepServiceRequest):
        pack = DeepServicePack()
        pack.metadata = MetadataUtils.struct_to_dict(request.metadata)
        pack.arrays, pack.arrays_action = DTensorUtils.dtensor_bundle_to_numpy(request.bundle)
        return pack


class SimpleDeepServiceClient(DeepServiceClient):

    def __init__(self, host='localhost', port=50051, cfg=DeepServiceCFG()):
        super(SimpleDeepServiceClient, self).__init__(host=host, port=port, cfg=cfg)

    def deep_serve(self, pack: DeepServicePack):
        request = pack.to_request()
        response: DDeepServiceResponse = self.DeepServe(request)
        reply_pack = DeepServicePack.from_deep_service_response(response)
        return reply_pack
