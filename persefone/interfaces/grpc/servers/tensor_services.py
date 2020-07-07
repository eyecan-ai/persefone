from persefone.interfaces.grpc.tensor_services_pb2_grpc import SimpleTensorServiceServicer, add_SimpleTensorServiceServicer_to_server
from persefone.interfaces.grpc.clients.tensor_services import MetaImagesServiceClient
from persefone.interfaces.proto.data_pb2 import DTensorBundle
from persefone.interfaces.proto.utils.dtensor import DTensorUtils
import threading
import grpc
from concurrent import futures
from typing import Callable
import json


class TensorServiceServerCFG(object):
    DEFAULT_MAX_MESSAGE_LENGTH = -1

    def __init__(self):
        self.options = [
            ('grpc.max_send_message_length', self.DEFAULT_MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', self.DEFAULT_MAX_MESSAGE_LENGTH),
        ]


class SimpleTensorServer(SimpleTensorServiceServicer):

    def __init__(self, consume_callback: Callable, host='0.0.0.0', port=50051, max_workers=10, options=TensorServiceServerCFG().options):
        self._consume_callback = consume_callback
        assert self._consume_callback is not None, "Server callback must be a valid callable function"
        self._host = host
        self._port = port
        self._options = options
        self._max_workers = max_workers
        self._server = None
        self._server_thread = None
        self._started = False
        self._failed_connection = False

    @property
    def active(self) -> bool:
        return self._started

    def start(self):
        self._server_thread = threading.Thread(target=self._serve, daemon=True)
        self._server_thread.start()

    def wait_for_termination(self):
        if self._server_thread is not None:
            self._server_thread.join()

    def stop(self):
        if self._server is not None:
            self._server.stop(0)
            self._started = False

    def _serve(self):
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=self._max_workers), options=self._options)
        add_SimpleTensorServiceServicer_to_server(self, self._server)
        port = self._server.add_insecure_port(f'{self._host}:{self._port}')
        if port == 0:
            raise Exception("GRPC Port is not valid!")
            self._failed_connection = True
            return False
        self._server.start()
        self._started = True
        self._server.wait_for_termination()
        return True

    def Consume(self, bundle: DTensorBundle, context: grpc.ServicerContext) -> DTensorBundle:
        repl_bundle = bundle
        return self._consume_callback(repl_bundle, context)


class MetaImagesTensorServer(SimpleTensorServer):

    def __init__(self, user_callback: Callable, host='0.0.0.0', port=50051, max_workers=10, options=TensorServiceServerCFG().options):
        super(MetaImagesTensorServer, self).__init__(
            consume_callback=self._raw_consume_callback,
            host=host,
            port=port,
            max_workers=max_workers,
            options=options
        )
        self._user_callback = user_callback

    def _raw_consume_callback(self, bundle: DTensorBundle, context: grpc.ServicerContext) -> DTensorBundle:

        # Converts protobuf DTensorBundle to list of numpy arrays with action string
        images, action = DTensorUtils.dtensor_bundle_to_numpy(bundle)

        # convert action string (JSON) to dictionary
        metadata = json.loads(action)

        # Call user callback
        reply_images, reply_metadata = self._user_callback(images, metadata)

        # converts user metadata to plain json string
        reply_action = json.dumps(reply_metadata)

        # Converts back to protobuf DTensorBundle
        reply_bundle = DTensorUtils.numpy_to_dtensor_bundle(reply_images, reply_action)
        return reply_bundle
