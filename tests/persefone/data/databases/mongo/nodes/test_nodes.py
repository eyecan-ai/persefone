from persefone.data.databases.mongo.nodes.nodes import MLink, MNode, NodesPath
import pytest
import numpy as np
import torch


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
            ('realm', True),
            ('realm///', False),
            ('realm//jiuojiojsad', False),
            ('adasdasda/jiuojiojsad/', True),
            ('realm/category/one', True),
            ('realm/category/one_special', True),
            ('realm/category/one_special/', True),
            ('realm/category/one_special/two/three', True),
            ('realm/category/one_special/two/three/4/1/2.2/hello/$2$', True),
            ('realm/category/one_special/two//three/4/1/2.2/hello/$2$', False),
        ]

    def test_path_separators(self, node_paths):

        for p, should_be in node_paths:

            np = NodesPath(p)
            assert np.valid is should_be, f"Node path {p} should be {'valid' if should_be else 'invalid'}"

            if not p.startswith(NodesPath.PATH_SEPARATOR):
                np_with_start = NodesPath(f'{NodesPath.PATH_SEPARATOR}{p}')
            #     #print("Adding to ", p, np_with_start, np_with_start.valid, np_with_start.value)
                assert np_with_start.valid is should_be, f"Node path {p} with added Separator should be {'valid' if should_be else 'invalid'}"

            if not p.endswith(NodesPath.PATH_SEPARATOR):
                np_with_start = NodesPath(f'{p}{NodesPath.PATH_SEPARATOR}')
            #     #print("Adding to ", p, np_with_start, np_with_start.valid, np_with_start.value)
                assert np_with_start.valid is should_be, f"Node path {p} with added Separator (end) should be {'valid' if should_be else 'invalid'}"

            if np.valid:
                assert not np.value.startswith(NodesPath.PATH_SEPARATOR), "No separator in front!"
                assert not np.value.endswith(NodesPath.PATH_SEPARATOR), "No separator in tail!"

            if should_be:
                np2 = NodesPath.builds_path(
                    *np.items
                )
                assert np.value == np2.value, "Paths values does not match!"

            if should_be:
                assert np.parent_path.value in np.value, "Parent path should be contained in child path"

                pieces = [np]
                parent: NodesPath = np.parent_path
                infinite_loop_watch = 1000
                while parent.valid:
                    pieces.append(parent)
                    parent = parent.parent_path
                    infinite_loop_watch -= 1
                    if infinite_loop_watch <= 0:
                        break
                assert not infinite_loop_watch < 0, "Recurisve loop detected! "

                assert len(pieces) == len(np.items), "Parents and items must match in size"
                assert len(pieces) == len(np.subpaths()), "Subpaths must match in size"

                for subpath in np.subpaths():
                    assert all(x in np.items for x in subpath.items), "Some item is missing in subpaths"
            print("==")

    @pytest.mark.mongo_real_server  # EXECUTE ONLY IF --mongo_real_server option is passed
    def test_nodes(self, temp_mongo_database):

        n_nodes = 32
        nodes_even = []
        nodes_odd = []
        for i in range(n_nodes):
            n = i * 2

            # Evens
            node_e = MNode.create(name=f'/realm/even/{n}')
            assert node_e is not None, "Node creation should be ok"
            node_e.metadata_ = {
                'number': n,
                'type': 'even'
            }
            node_e.save()
            nodes_even.append(node_e)

            # Odds
            node_o = MNode.create(name=f'/realm/odds/{n}')
            assert node_o is not None, "Node creation should be ok"
            node_o.metadata_ = {
                'number': n + 1,
                'type': 'odd'
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
            assert len(node.outbound_nodes()) == 0, "Event -> Odd links must be empty"

        # Checks for Odd -> Even links non empty
        for node in nodes_odd:
            node: MNode
            assert len(node.outbound()) > 0, "Event -> Odd links must be not empty"
            for link in node.outbound():
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
    # def test_nodes_realm(self, temp_mongo_persistent_database: MongoDatabaseClient):

    #     NR = NodesRealm(client_cfg=temp_mongo_persistent_database.cfg)

    #     from pathlib import Path

    #     print(NR[NR.realm / 'cat' / 'object'])
    #     print(NR.get_node_by_chunks(NR.realm, 'cat', 'object'))
    #     print(NR)
