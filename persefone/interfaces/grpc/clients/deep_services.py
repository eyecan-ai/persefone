from persefone.interfaces.proto.utils.dtensor import DTensorUtils
from persefone.interfaces.proto.utils.comm import MetadataUtils
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
        self.arrays = {}
        self.arrays_action = ''


class SimpleDeepServiceClient(DeepServiceClient):

    def __init__(self, host='localhost', port=50051, cfg=DeepServiceCFG()):
        super(SimpleDeepServiceClient, self).__init__(host=host, port=port, cfg=cfg)

    def deep_serve(self, pack: DeepServicePack):

        request = DDeepServiceRequest()
        request.bundle.CopyFrom(DTensorUtils.numpy_to_dtensor_bundle(pack.arrays, pack.arrays_action))
        MetadataUtils.dict_to_struct(pack.metadata, request.metadata)

        response: DDeepServiceResponse = self.DeepServe(request)

        reply_pack = DeepServicePack()
        reply_pack.metadata = MetadataUtils.struct_to_dict(response.metadata)
        reply_pack.arrays, reply_pack.arrays_action = DTensorUtils.dtensor_bundle_to_numpy(response.bundle)

        return reply_pack
