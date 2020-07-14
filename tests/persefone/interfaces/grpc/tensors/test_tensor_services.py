
from persefone.interfaces.grpc.clients.tensor_services import SimpleTensorServiceClient, TensorServiceClientCFG, MetaImagesServiceClient
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

    def __init__(self, port=50051, decorate_action=True, options=TensorServiceClientCFG().options):
        self._port = port
        self._options = options
        self._server = None
        self._started = False
        self._decorate_action = decorate_action

    @property
    def active(self):
        return self._started

    def start(self, decorate_action=True):
        self._decorate_action = decorate_action
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
        if self._decorate_action:
            repl_bundle.action = self.build_fake_action_reply(bundle.action)
        return repl_bundle

    @classmethod
    def build_fake_action_reply(cls, action):
        return '#REPLY#_' + action + "_#!!#"


@pytest.fixture(scope='function')
def mock_server():
    return MockSimpleTensorServiceServicer()


class TestSimpleTensorServiceClient(object):

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
                print(f"Sending data size [{numpy_dtype}]: {bundle.ByteSize()}")

                reply_bundle = client.consume(bundle)
                print(f"\t Received data size: {reply_bundle.ByteSize()}")
                reply_arrays, reply_action = DTensorUtils.dtensor_bundle_to_numpy(reply_bundle)
                assert reply_action == MockSimpleTensorServiceServicer.build_fake_action_reply(bundle.action), "Action is wrong!"
                assert len(reply_arrays) == len(arrays_to_send), "Reply arrays number is wrong!"
                for idx in range(len(reply_arrays)):
                    a, b = arrays_to_send[idx], reply_arrays[idx]
                    assert np.array_equal(a, b), "Replay Arrays content is wrong!"
                    assert a.shape == b.shape, "Reply arrays shape is wrong!"
                    assert a.dtype == b.dtype, "Reply arrays dtype is wrong!"

        mock_server.stop()


class TestMetaImagesServiceClient(object):

    @pytest.fixture()
    def testing_images(self):
        return [
            {'dtype': np.uint8, 'shape': (3, 600, 800)},
            {'dtype': np.float, 'shape': (3, 256, 256)},
            {'dtype': np.uint16, 'shape': (256, 256, 1)},
        ]

    @pytest.fixture()
    def testing_metadata(self):
        return [
            {'action': 'inference', 'batch_size': 16, 'quantization': False},
            {'learning_rate': 0.001, 'loss_multipliers': [1.5, 2.5, 3.5]},
            {'optimizer_params': {'alpha': 0.99, 'beta': 0.001}},
        ]

    def _compare_param(self, p1, p2):
        if isinstance(p1, list) or isinstance(p1, tuple):
            return np.all(np.isclose(np.array(p1), np.array(p2)))
        else:
            return p1 == p2

    def test_service_online(self, mock_server, testing_images, testing_metadata):
        """ Tests an Online service sending multiple bundles """

        mock_server.start(decorate_action=False)

        while not mock_server.active:
            time.sleep(0.1)

        # client end point
        client = MetaImagesServiceClient()

        images_to_send = []

        for image_cfg in testing_images:

            img = np.array(np.random.uniform(-100, 100, image_cfg['shape']), dtype=image_cfg['dtype'])
            images_to_send.append(img)

            for metadata in testing_metadata:

                reply_images, reply_meta = client.send_meta_images(images_to_send, metadata)
                assert len(reply_images) == len(images_to_send), "Number of reply images is wrong!"

                print("Reply metadata: ", reply_meta)
                for k, v in metadata.items():
                    assert k in reply_meta, f"Missing metadata key: {k}"
                    assert self._compare_param(metadata[k], reply_meta[k]), f"Value of '{k}' is different in reply metadata"

        mock_server.stop()
