
from deepdiff import DeepDiff
from persefone.data.databases.mongo.nodes.nodes import NodesRealm
from persefone.data.databases.mongo.clients import MongoDatabaseClient
import time
import pytest
import numpy as np
import pickle
import torch
from itertools import product
from tqdm import tqdm


class TestNodesManagement(object):

    @pytest.fixture
    def node_data(self):
        return [
            np.random.uniform(-1, 1, (100, 100, 100)).astype(np.float32),
            np.random.uniform(-1, 1, (4, 5, 1)).astype(np.float64),
            torch.Tensor(np.random.uniform(-1, 1, (4, 5, 1)).astype(np.float64)),
            {'a': 2.2, 'b': 3.3, 'c': 2.4},
            {'a': 2.2, 'b': 'bye', 'c': {'one': [2, [2], 3]}},
            "this is a string",
            2.2,
            2,
            True
        ]

    @pytest.mark.mongo_real_server  # EXECUTE ONLY IF --mongo_real_server option is passed
    def test_nodes_creation(self, temp_mongo_persistent_database: MongoDatabaseClient, node_data):

        NR = NodesRealm(mongo_client=temp_mongo_persistent_database)

        for idx, data in enumerate(node_data):
            name = f'nm.node_{idx}'
            NR[name].data = data

        for idx, data in enumerate(node_data):
            name = f'nm.node_{idx}'
            retrieved_data = NR[name].data
            if isinstance(retrieved_data, np.ndarray):
                assert np.array_equal(data, retrieved_data)
            elif isinstance(retrieved_data, torch.Tensor):
                assert data.equal(retrieved_data)
            elif isinstance(retrieved_data, dict):
                ddiff = DeepDiff(data, retrieved_data, ignore_order=True, ignore_numeric_type_changes=True)
                assert not ddiff
            else:
                assert data == retrieved_data

            NR[name].delete()

    @pytest.mark.mongo_real_server  # EXECUTE ONLY IF --mongo_real_server option is passed
    def test_nodes_links(self, temp_mongo_persistent_database: MongoDatabaseClient):

        NR = NodesRealm(mongo_client=temp_mongo_persistent_database)

        import cv2

        for node in NR.get_nodes_by_category('sample'):
            print(node.namespace, node.name)
            for k, item in node.links.items():
                data = item.data

                cv2.imshow("image", data)
                cv2.waitKey(1)
        # dataset_name = 'pino'
        # n_samples = 1000
        # n_items = 3

        # NR[dataset_name].category = 'dataset'

        # for sample_id in tqdm(range(n_samples)):
        #     print(sample_id)

        #     sample_name = f'{dataset_name}_sample_{str(sample_id).zfill(5)}'

        #     NR[sample_name].category = 'sample'
        #     NR[sample_name].data = {'a': 2.2}
        #     NR[dataset_name].link_to(NR[sample_name], sample_name)

        #     for item_id in range(n_items):

        #         item_name = f'{sample_name}_item_{str(item_id).zfill(5)}'
        #         NR[item_name].category = 'item'
        #         NR[item_name].data = np.random.uniform(0, 255, (512, 512, 3)).astype(np.uint8)
        #         NR[sample_name].link_to(NR[item_name], item_name)

            # items = range(32)
            # evens = [x for x in items if x % 2 == 0]
            # odds = [x for x in items if x % 2 == 1]
            # fives = [x for x in items if x % 5 == 0 and x > 0]

            # p_evens = product(evens, evens)

            # for first, second in p_evens:
            #     if first != second:
            #         NR[first].data = np.random.uniform(0, 255, (512, 512, 3)).astype(np.uint8)
            #         NR[first].link_to(NR[second], f'{second}')
            #         NR[first].category = 'even'
            #         NR[second].category = 'even'
