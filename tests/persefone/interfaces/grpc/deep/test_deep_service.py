
from persefone.interfaces.grpc.servers.deep_services import EchoDeepService
from persefone.interfaces.grpc.clients.deep_services import DeepServiceCFG, DeepServicePack, SimpleDeepServiceClient
import grpc
from concurrent import futures
import threading
from deepdiff import DeepDiff
import numpy as np
import pytest


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

    def test_lifecycle(self,  random_metadata):

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

        # Self repacking
        repack_0 = pack.from_deep_service_response(pack.to_response())
        repack_1 = pack.from_deep_service_request(pack.to_request())
        for repack in [repack_0, repack_1]:
            assert not DeepDiff(pack.metadata, repack.metadata, ignore_order=True, ignore_numeric_type_changes=True)
            # Check numerical data equality
            assert len(repack.arrays) == len(pack.arrays), "Number of retrieved arrays is wrong!"
            for idx in range(len(repack.arrays)):
                assert np.array_equal(pack.arrays[idx], repack.arrays[idx]), f"Array {idx} is different!"

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
