

from persefone.data.databases.mongo.nodes.nodes import MLink, MNode, NodesBucket
from mongoengine.errors import DoesNotExist
from persefone.data.databases.mongo.clients import MongoDatabaseClientCFG


class DatasetsBucket(NodesBucket):

    DATASET_NAMSPACE_NAME = '$DATASETS'
    NODE_TYPE_DATASET = 'dataset'
    NODE_TYPE_SAMPLE = 'sample'
    NODE_TYPE_ITEM = 'item'
    LINK_TYPE_DATASET2SAMPLE = 'dataset_2_sample'
    LINK_TYPE_SAMPLE2ITEM = 'sample_2_item'
    DEFAULT_SAMPLE_ID_ZERO_PADDING = 7

    def __init__(self, client_cfg: MongoDatabaseClientCFG, namespace: str = None):
        super(DatasetsBucket, self).__init__(client_cfg, self.DATASET_NAMSPACE_NAME)

    def get_datasets(self):
        return self.get_namespace_node().outbound_nodes(link_type=self.LINK_TYPE_NAMESPACE2GENERIC)

    def new_dataset(self, dataset_name):

        try:
            self.get_dataset(dataset_name)
            raise NameError(f"Dataset with same name '{dataset_name}' already exists")

        except DoesNotExist:

            namespace_node: MNode = self.get_namespace_node()

            dataset_node: MNode = self[self.namespace / dataset_name]
            dataset_node.node_type = self.NODE_TYPE_DATASET

            namespace_node.link_to(dataset_node, link_type=self.LINK_TYPE_NAMESPACE2GENERIC)
            return dataset_node

    def get_dataset(self, dataset_name: str) -> MNode:
        """ Get dataset node by name

        :param dataset_name: dataset name
        :type dataset_name: str
        :return: retrieved MNode
        :raises DoesNotExist: raises Exception if related node does not exists
        :return: retrieved MNode
        :rtype: MNode
        """

        return self.get_node_by_name(self.namespace / dataset_name)

    def get_samples(self, dataset_name: str):
        dataset_node: MNode = self.get_dataset(dataset_name)
        return dataset_node.outbound_nodes(link_type=self.LINK_TYPE_DATASET2SAMPLE)

    def get_sample(self, dataset_name: str, sample_id: str):
        return self.get_node_by_name(self.namespace / dataset_name / self._sample_id_name(sample_id))

    def get_item(self, dataset_name: str, sample_id: int, item_name: str):
        return self.get_node_by_name(self.namespace / dataset_name / self._sample_id_name(sample_id) / item_name)

    def get_items(self, dataset_name: str, sample_id: int):
        sample_node: MNode = self.get_sample(dataset_name, sample_id)
        return sample_node.outbound_nodes(link_type=self.LINK_TYPE_SAMPLE2ITEM)

    def delete_dataset(self, dataset_name):
        dataset_node = self.get_dataset(dataset_name)
        samples = dataset_node.outbound_nodes(link_type=self.LINK_TYPE_DATASET2SAMPLE)
        for sample_node in samples:
            items = sample_node.outbound_nodes(link_type=self.LINK_TYPE_SAMPLE2ITEM)
            for item_node in items:
                item_node.delete()
            sample_node.delete()
        dataset_node.delete()

    def _sample_id_name(self, sample_id: int):
        return str(sample_id).zfill(self.DEFAULT_SAMPLE_ID_ZERO_PADDING)

    def new_sample(self, dataset_name, metadata: dict = None, sample_id: int = -1):

        dataset_node: MNode = self.get_dataset(dataset_name)

        if sample_id < 0:
            samples = MLink.outbound_of(dataset_node, self.LINK_TYPE_DATASET2SAMPLE)
            n_samples = len(samples)
            sample_id = n_samples

        try:
            self.get_sample(dataset_name, self._sample_id_name(sample_id))
            raise NameError(f"Sample with sample id '{sample_id}' was found")
        except DoesNotExist:

            sample_node = self[self.namespace / dataset_name / self._sample_id_name(sample_id)]
            metadata['_sample_id'] = sample_id
            sample_node.metadata_ = metadata
            sample_node.node_type = self.NODE_TYPE_SAMPLE
            dataset_node.link_to(sample_node, link_type=self.LINK_TYPE_DATASET2SAMPLE)
            return sample_node

    def new_item(self, dataset_name: str, sample_id: int, item_name: str, blob_data: bytes = None, blob_encoding: str = None):

        sample_node: MNode = self.get_sample(dataset_name, sample_id)

        try:
            self.get_item(dataset_name, sample_id, item_name)
            raise NameError("Item with same name '{item_name}' found")
        except DoesNotExist:

            item_node: MNode = self[self.namespace / dataset_name / self._sample_id_name(sample_id) / item_name]

            sample_node.link_to(item_node, link_type=self.LINK_TYPE_SAMPLE2ITEM)
            item_node.node_type = self.NODE_TYPE_ITEM
            item_node.put_data(blob_data, blob_encoding)
            return item_node
