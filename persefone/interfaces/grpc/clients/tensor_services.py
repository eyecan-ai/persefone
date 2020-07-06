
from persefone.interfaces.grpc.tensor_services_pb2_grpc import SimpleTensorServiceStub
from persefone.interfaces.proto.data_pb2 import DTensorBundle
import grpc


class SimpleTensorServiceClientCFG(object):
    DEFAULT_MAX_MESSAGE_LENGTH = -1

    def __init__(self):
        self.options = [
            ('grpc.max_send_message_length', self.DEFAULT_MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', self.DEFAULT_MAX_MESSAGE_LENGTH),
        ]


class SimpleTensorServiceClient(object):

    def __init__(self, host='localhost', port=50051, cfg=SimpleTensorServiceClientCFG()):
        if isinstance(port, str):
            port = int(port)
        self._channel = grpc.insecure_channel(f'{host}:{port}', options=cfg.options)
        self._stub = SimpleTensorServiceStub(self._channel)

    def consume(self, bundle: DTensorBundle) -> DTensorBundle:
        """ GRPC -> Consume endpoint

        :param bundle: request protobuf DTensorBundle object
        :type bundle: DTensorBundle
        :return: protobuf DTensorBundle reply
        :rtype: DTensorBundle
        """

        return self._stub.Consume(bundle)
