
from persefone.interfaces.grpc.clients.tensor_services import SimpleTensorServiceClient, SimpleTensorServiceClientCFG
from persefone.interfaces.grpc.tensor_services_pb2_grpc import SimpleTensorServiceServicer, add_SimpleTensorServiceServicer_to_server
from persefone.interfaces.proto.data_pb2 import DTensorBundle
from persefone.interfaces.proto.utils.dtensor import DTensorUtils

import numpy as np
import pytest
import threading
import grpc
import concurrent.futures as futures
import time
import random


class MockSimpleTensorServiceServicer(SimpleTensorServiceServicer):

    def __init__(self, port=50051, options=SimpleTensorServiceClientCFG().options):
        self._port = port
        self._options = options
        self._server = None
        self._started = False

    @property
    def active(self):
        return self._started

    def start(self):
        t = threading.Thread(target=self._serve, daemon=True)
        t.start()

    def stop(self):
        if self._server is not None:
            self._server.stop(0)
            self._started = False

    def _serve(self):
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=self._options)
        add_SimpleTensorServiceServicer_to_server(self, self._server)
        self._server.add_insecure_port(f'[::]:{self._port}')
        self._server.start()
        self._started = True
        self._server.wait_for_termination()

    def Consume(self, bundle: DTensorBundle, context):
        repl_bundle = bundle
        repl_bundle.action = self.build_fake_action_reply(bundle.action)
        return repl_bundle

    @classmethod
    def build_fake_action_reply(cls, action):
        return '#REPLY#_' + action + "_#!!#"


class TestSimpleTensorServiceClient(object):

    @pytest.fixture(scope='function')
    def mock_server(self):
        return MockSimpleTensorServiceServicer()

    @pytest.fixture()
    def testing_arrays(self):
        return [
            {'size': 3 * 600 * 800, 'shape': (3, 600, 800)},
            {'size': 3 * 600 * 800, 'shape': (3 * 600, 800)},
            {'size': 3 * 600 * 800, 'shape': (3 * 600 * 800)},
        ]

    def test_service_offline(self, testing_arrays):
        """ Tests an offline server call, should raise an Exception """

        client = SimpleTensorServiceClient()
        bundle = DTensorUtils.numpy_to_dtensor_bundle([], "fake_command")

        with pytest.raises(Exception):
            reply_bundle = client.consume(bundle)
            print(reply_bundle)

    def test_service_online(self, mock_server, testing_arrays):
        """ Tests an Online service sending multiple bundles """

        mock_server.start()

        while not mock_server.active:
            time.sleep(0.1)

        client = SimpleTensorServiceClient()

        for test_array in testing_arrays:
            for numpy_dtype, _ in DTensorUtils.NUMPY_TYPES_MAPPING.items():
                x = np.array(np.random.uniform(-100, 100, test_array['shape']), dtype=np.dtype(numpy_dtype))

                arrays_to_send = [x, x, x]

                bundle = DTensorUtils.numpy_to_dtensor_bundle(arrays_to_send, "fake_command")
                print(f"Sending bundle of size {numpy_dtype}: {bundle.ByteSize()}")

                reply_bundle = client.consume(bundle)
                reply_arrays, reply_action = DTensorUtils.dtensor_bundle_to_numpy(reply_bundle)
                assert reply_action == MockSimpleTensorServiceServicer.build_fake_action_reply(bundle.action), "Action is wrong!"
                assert len(reply_arrays) == len(arrays_to_send), "Reply arrays number is wrong!"
                for idx in range(len(reply_arrays)):
                    a, b = arrays_to_send[idx], reply_arrays[idx]
                    assert np.array_equal(a, b), "Replay Arrays content is wrong!"
                    assert a.shape == b.shape, "Reply arrays shape is wrong!"
                    assert a.dtype == b.dtype, "Reply arrays dtype is wrong!"

        mock_server.stop()
