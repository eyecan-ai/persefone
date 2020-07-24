
from persefone.interfaces.proto.utils.comm import MetadataUtils
import deepdiff
from persefone.interfaces.grpc.deep_services_pb2 import DDeepServiceRequest, DDeepServiceResponse
import pytest
from persefone.data.io.drivers.safefs import SafeFilesystemDriver, SafeFilesystemDriverCFG
from persefone.interfaces.grpc.servers.deep_services import DeepService, EchoDeepService, MongoDeepService
from persefone.interfaces.grpc.clients.deep_services import DeepServiceClient, DeepServiceCFG, DeepServicePack, SimpleDeepServiceClient
from persefone.utils.bytes import DataCoding
import grpc
from google.protobuf import json_format
from concurrent import futures
import threading
from deepdiff import DeepDiff
import numpy as np


class TestEchoDeepService(object):

    @pytest.fixture
    def random_metadata(self):
        return {
            'one': 1,
            'two': [[[0], [2], [3]]],
            'three': 3.3,
            '4': 'four',
            'dd': {
                'a': 1.1,
                'b': True,
                'c': {
                    'c.1': 0000,
                    'c.2': 'c1',
                    'c.3': [1, 2, 3, 4, 5, 5, .33]
                }
            }
        }

    @pytest.mark.mongo_real_server  # EXECUTE ONLY IF --mongo_real_server option is passed
    def test_lifecycle(self, temp_mongo_database, driver_temp_base_folder, random_metadata):
        self._test_lifecycle(temp_mongo_database, driver_temp_base_folder, random_metadata)

    # @pytest.mark.mongo_mock_server
    # def test_lifecycle_mock(self, temp_mongo_mock_database, driver_temp_base_folder, minimnist_folder):
    #     self._test_lifecycle(temp_mongo_mock_database, driver_temp_base_folder, minimnist_folder)

    def _test_lifecycle(self, mongo_client, driver_temp_base_folder, random_metadata):

        host = 'localhost'
        port = 10005

        service = EchoDeepService()

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=DeepServiceCFG().options)
        server.add_insecure_port(f'{host}:{port}')

        service.register(server)

        def _serve():
            server.start()
            server.wait_for_termination()

        t = threading.Thread(target=_serve, daemon=True)
        t.start()

        client = SimpleDeepServiceClient(host=host, port=port)

        # Pack to Send
        pack = DeepServicePack()
        pack.arrays = [
            np.random.uniform(0, 1, (256, 256, 3)).astype(np.uint8),
            np.random.uniform(0, 1, (256, 256, 3)).astype(np.float32),
            np.random.uniform(0, 1, (256, 256, 3)).astype(np.int64)
        ]
        pack.arrays_action = 'my_Action'
        pack.metadata = random_metadata

        # gRPC
        reply_pack = client.deep_serve(pack)

        # Copmute MEtadata deep diff
        ddiff = DeepDiff(pack.metadata, reply_pack.metadata, ignore_order=True, ignore_numeric_type_changes=True)
        assert not ddiff, "Metadata transfer fails!"

        import pprint
        pprint.pprint(pack.metadata)
        print("=" * 20)
        pprint.pprint(reply_pack.metadata)
        print("=" * 20)
        print(ddiff)

        # Check numerical data equality
        assert len(reply_pack.arrays) == len(pack.arrays), "Number of retrieved arrays is wrong!"
        for idx in range(len(reply_pack.arrays)):
            assert np.array_equal(pack.arrays[idx], reply_pack.arrays[idx]), f"Array {idx} is different!"

        ##################################################
        # Service teardown
        ##################################################
        server.stop(grace=None)
        t.join()
