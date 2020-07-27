
from deepdiff import DeepDiff
from persefone.data.databases.mongo.nodes.nodes import MLink, MNode, NodesPath, NodesRealm
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

    @pytest.fixture
    def node_paths(self):
        return [
            ('', False),
            ('realm', False),
            ('realm///', False),
            ('realm//jiuojiojsad', False),
            ('adasdasda/jiuojiojsad/', False),
            ('realm/category/one', True),
            ('realm/category/one_special', True),
            ('realm/category/one_special/two/three', True),
            ('realm/category/one_special/two/three/4/1/2.2/hello/$2$', True),
        ]

    def test_path_separators(self, node_paths):

        for p, should_be in node_paths:

            np = NodesPath(p)
            print(np.items)
            assert np.valid is should_be, f"Node path {p} should be {'valid' if should_be else 'invalid'}"

            if not p.startswith(NodesPath.PATH_SEPARATOR):
                np_with_start = NodesPath(f'{NodesPath.PATH_SEPARATOR}{p}')
                assert np_with_start.valid is should_be, f"Node path {p} with added Separator should be {'valid' if should_be else 'invalid'}"

            if should_be:
                np2 = NodesPath.builds_path(
                    np.namespace,
                    np.category,
                    np.items
                )
                assert np.value == np2.value, "Paths values does not match!"

    @pytest.mark.mongo_real_server  # EXECUTE ONLY IF --mongo_real_server option is passed
    def test_nodes(self, temp_mongo_persistent_database):

        n_nodes = 32
        nodes_even = []
        nodes_odd = []
        for i in range(n_nodes):
            n = i * 2

            # Evens
            node_e = MNode(name_=f'/realm/even/{n}')
            node_e.metadata_ = {
                'number': n,
                'type': 'even'
            }
            node_e.save()
            nodes_even.append(node_e)

            # Odds
            node_o = MNode(name_=f'/realm/odds/{n + 1}')
            node_o.metadata_ = {
                'number': n + 1,
                'type': 'even'
            }
            node_o.save()
            nodes_odd.append(node_o)

        whole_nodes = nodes_even + nodes_odd

        for ne, no in zip(nodes_even, nodes_odd):
            ne: MNode
            no: MNode
            ne.link_to(no, metadata={'start': ne.metadata_['number']}, link_type='e2o')
            no.link_to(ne, metadata={'start': no.metadata_['number']}, link_type='o2e')

        o2e = MLink.objects(link_type_='o2e')
        e2o = MLink.objects(link_type_='e2o')
        assert len(o2e) == n_nodes, "number of o2e links is wrong"
        assert len(e2o) == n_nodes, "number of e2o links is wrong"

        # Delete links Even -> Odd
        for link in e2o:
            link: MLink
            assert link.link_type == 'e2o', "Link type is wrong!"
            link.delete()

        # Check links Odd -> Even
        for link in o2e:
            link: MLink
            assert link.link_type == 'o2e', "Link type is wrong!"

        # Checks for Even -> Odd links empty
        for node in nodes_even:
            node: MNode
            assert len(node.outbound) == 0, "Event -> Odd links must be empty"

        # Checks for Odd -> Even links non empty
        for node in nodes_odd:
            node: MNode
            assert len(node.outbound) > 0, "Event -> Odd links must be not empty"
            for link in node.outbound:
                assert link.start_node == node, "Something was wrong!!"
                with pytest.raises(AttributeError):  # fields of Lazy reference cannot be accessed
                    assert link.start_node.name == node.name
                assert link.start_node.fetch().name == node.name, "Name should be equal"

        # Delete all nodes
        for node in whole_nodes:
            node: MNode
            node.delete()

        assert len(MNode.objects()) == 0, "No nodes must be there!"
        assert len(MLink.objects()) == 0, "No nodes must be there!"

    # @pytest.mark.mongo_real_server  # EXECUTE ONLY IF --mongo_real_server option is passed
    # def test_nodes_creation(self, temp_mongo_persistent_database: MongoDatabaseClient, node_data):

    #     NR = NodesRealm(client_cfg=temp_mongo_persistent_database.cfg)

    #     for idx, data in enumerate(node_data):
    #         name = f'nm.node_{idx}'
    #         NR[name].data = data

    #     for idx, data in enumerate(node_data):
    #         name = f'nm.node_{idx}'
    #         print(name)
    #         retrieved_data = NR[name].data

    #         if isinstance(retrieved_data, np.ndarray):
    #             assert np.array_equal(data, retrieved_data)
    #         elif isinstance(retrieved_data, torch.Tensor):
    #             assert data.equal(retrieved_data)
    #         elif isinstance(retrieved_data, dict):
    #             ddiff = DeepDiff(data, retrieved_data, ignore_order=True, ignore_numeric_type_changes=True)
    #             assert not ddiff
    #         else:
    #             assert data == retrieved_data

    #         # NR[name].delete()

    # @pytest.mark.mongo_real_server  # EXECUTE ONLY IF --mongo_real_server option is passed
    # def test_nodes_links(self, temp_mongo_persistent_database: MongoDatabaseClient):

    #     NR = NodesRealm(mongo_client=temp_mongo_persistent_database)

    #     import cv2

    #     for node in NR.get_nodes_by_category('sample'):
    #         print(node.namespace, node.name)
    #         for k, item in node.links.items():
    #             data = item.data

    #             cv2.imshow("image", data)
    #             cv2.waitKey(1)
    #     # dataset_name = 'pino'
    #     # n_samples = 1000
    #     # n_items = 3

    #     # NR[dataset_name].category = 'dataset'

    #     # for sample_id in tqdm(range(n_samples)):
    #     #     print(sample_id)

    #     #     sample_name = f'{dataset_name}_sample_{str(sample_id).zfill(5)}'

    #     #     NR[sample_name].category = 'sample'
    #     #     NR[sample_name].data = {'a': 2.2}
    #     #     NR[dataset_name].link_to(NR[sample_name], sample_name)

    #     #     for item_id in range(n_items):

    #     #         item_name = f'{sample_name}_item_{str(item_id).zfill(5)}'
    #     #         NR[item_name].category = 'item'
    #     #         NR[item_name].data = np.random.uniform(0, 255, (512, 512, 3)).astype(np.uint8)
    #     #         NR[sample_name].link_to(NR[item_name], item_name)

    #         # items = range(32)
    #         # evens = [x for x in items if x % 2 == 0]
    #         # odds = [x for x in items if x % 2 == 1]
    #         # fives = [x for x in items if x % 5 == 0 and x > 0]

    #         # p_evens = product(evens, evens)

    #         # for first, second in p_evens:
    #         #     if first != second:
    #         #         NR[first].data = np.random.uniform(0, 255, (512, 512, 3)).astype(np.uint8)
    #         #         NR[first].link_to(NR[second], f'{second}')
    #         #         NR[first].category = 'even'
    #         #         NR[second].category = 'even'
