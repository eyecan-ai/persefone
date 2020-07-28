

from os import link

import logging
from persefone.data.databases.mongo.snapshots import SnapshotOperations
from schema import Optional, Schema
from persefone.utils.configurations import XConfiguration
from persefone.utils.mongo.queries import MongoQueryParser
from typing import List
from persefone.data.databases.mongo.nodes.nodes import MLink, MNode, NodesBucket
from mongoengine.errors import DoesNotExist
from persefone.data.databases.mongo.clients import MongoDatabaseClientCFG
from persefone.utils.bytes import DataCoding


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
            dataset_node.save()

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

    def count_samples(self, dataset_name: str):
        return len(self.get_samples(dataset_name))

    def get_sample(self, dataset_name: str, sample_id: str):
        return self.get_node_by_name(self.namespace / dataset_name / self._sample_id_name(sample_id))

    def get_samples_by_query(self, dataset_name: str, queries: list = None, orders_by: list = None):
        if queries is None:
            queries = []
        if orders_by is None:
            orders_by = []

        orders_by = orders_by.copy()
        queries = queries.copy()
        queries.append(f"node_type == '{self.NODE_TYPE_SAMPLE}'")
        queries.append(f"metadata.#dataset_name == '{dataset_name}'")
        query_dict = MongoQueryParser.parse_queries_list(queries)
        orders_bys = MongoQueryParser.parse_orders_list(orders_by)
        return MNode.get_by_queries(query_dict, orders_bys)

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

            sample_node: MNode = self[self.namespace / dataset_name / self._sample_id_name(sample_id)]
            metadata['#sample_id'] = sample_id
            metadata['#dataset_name'] = dataset_name
            sample_node.metadata = metadata
            sample_node.node_type = self.NODE_TYPE_SAMPLE
            sample_node.save()
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

    @classmethod
    def remap_sample_with_items(cls, sample: MNode, data_mapping: dict):

        output_data = {}

        # Fetches plain data
        metadata = sample.metadata

        for k, v in metadata.items():
            extended_k = f'metadata.{k}'
            if extended_k in data_mapping:
                output_data[data_mapping[extended_k]] = v

        # Fetches Items
        items = MLink.outbound_nodes_of(sample, link_type=cls.LINK_TYPE_SAMPLE2ITEM)
        for item in items:
            item: MNode
            extended_name = f'items.{item.last_name}'
            if extended_name in data_mapping:
                data, encoding = item.get_data()
                output_data[data_mapping[extended_name]] = DataCoding.bytes_to_data(data, encoding)

        return output_data


class DatasetsBucketReaderCFG(XConfiguration):

    def __init__(self, filename=None):
        XConfiguration.__init__(self, filename=filename)

        self.set_schema(Schema({
            # DATA MAPPING
            'data_mapping': {str: str},
            # QUERIES
            Optional('queries'): [str],
            # ORDERS
            Optional('orders'): [str],
        }))


class DatasetsBucketReader(object):

    def __init__(self, datasets_bucket: DatasetsBucket,
                 dataset_name: str,
                 data_mapping: dict = {},
                 queries: List[str] = [],
                 orders: List[str] = []):

        self._data_mapping = data_mapping
        self._datasets_bucket = datasets_bucket
        self._samples = list(self._datasets_bucket.get_samples_by_query(
            dataset_name=dataset_name,
            queries=queries,
            orders_by=orders
        ))
        self._queries = queries
        self._orders = orders

    @property
    def orders(self):
        return self._orders

    @property
    def queries(self):
        return self._queries

    @property
    def data_mapping(self):
        return self._data_mapping

    @property
    def samples(self):
        return self._samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        if idx > len(self):
            raise IndexError
        sample = self.samples[idx]
        return DatasetsBucket.remap_sample_with_items(sample, self._data_mapping)

    @classmethod
    def builds_from_configuration_file(cls, bucket: DatasetsBucket, dataset_name: str, filename):
        cfg = DatasetsBucketReaderCFG(filename=filename)
        return DatasetsBucketReader(
            bucket,
            dataset_name,
            data_mapping=cfg.params.data_mapping,
            queries=cfg.params.get('queries', []),
            orders=cfg.params.get('orders', [])
        )

    @classmethod
    def builds_from_configuration_dict(cls, bucket: DatasetsBucket, dataset_name: str, cfg: dict):
        cfg = DatasetsBucketReaderCFG.from_dict(cfg)
        return DatasetsBucketReader(
            bucket,
            dataset_name,
            data_mapping=cfg.params.data_mapping,
            queries=cfg.params.get('queries', []),
            orders=cfg.params.get('orders', [])
        )


class DatasetsBucketSamplesListReader(object):

    def __init__(self, samples: List[MNode], data_mapping: dict):
        self._samples = samples
        self._data_mapping = data_mapping

    @property
    def samples(self):
        return self._samples

    @property
    def data_mapping(self):
        return self._data_mapping

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        if idx > len(self):
            raise IndexError
        sample = self.samples[idx]
        return DatasetsBucket.remap_sample_with_items(sample, self._data_mapping)


class DatasetsBucketIsolatedSample(object):

    def __init__(self, sample: MNode, data_mapping: dict):
        self.sample = sample
        self.data_mapping = data_mapping


class DatasetsBucketIsolatedSamplesListReader(object):

    def __init__(self, isolated_samples: List[DatasetsBucketIsolatedSample]):
        self._isolated_samples = isolated_samples

    @property
    def isolated_samples(self):
        return self._isolated_samples

    def __len__(self):
        return len(self.isolated_samples)

    def __getitem__(self, idx):
        if idx > len(self):
            raise IndexError
        isolated_sample = self.isolated_samples[idx]
        return DatasetsBucket.remap_sample_with_items(isolated_sample.sample, isolated_sample.data_mapping)


class DatasetsBucketSnapshotCFG(XConfiguration):

    def __init__(self, filename=None):
        XConfiguration.__init__(self, filename=filename)
        self.set_schema(Schema({
            # NAME
            Optional('_reusable'): dict,
            'datasets': [
                {str: {
                    'dataset': {'name': str},
                    'reader': {'data_mapping': {str: str},
                               Optional('queries'): [str],
                               Optional('orders'): [str]
                               }
                }
                }
            ],
            'pipeline': str
        }))


class DatasetsBucketSnapshot(object):

    def __init__(self, bucket: DatasetsBucket, cfg: DatasetsBucketSnapshotCFG):

        self._cfg = cfg
        self._cfg.validate()
        self._bucket = bucket
        self._readers_map = {}

        for dataset in cfg.params.datasets:
            reader_name = list(dataset.keys())[0]
            dataset_name = dataset[reader_name]['dataset']['name']

            reader = DatasetsBucketReader.builds_from_configuration_dict(
                self._bucket,
                dataset_name,
                dataset[reader_name]['reader']
            )

            self._readers_map[reader_name] = reader

        # Generates a TrampSample map befor spuffle
        self._samples_map = {}
        for reader_name, reader in self._readers_map.items():
            self._samples_map[reader_name] = []
            reader: DatasetsBucketReader
            for sample in reader.samples:

                self._samples_map[reader_name].append(
                    DatasetsBucketIsolatedSample(sample, reader.data_mapping)
                )

        # Tries to load custom generate_output_list function from configuration file
        try:
            exec(self._cfg.params.pipeline)
            self.__class__.generate_output_lists = locals()['generate']
            output_data = self.generate_output_lists(SnapshotOperations, self._samples_map)
        except Exception as e:
            logging.error(e)
            output_data = self._samples_map
            raise NotImplementedError()

        # Transforms output data as a dict of MongoMixedSampleIterator
        self._output_data = {}
        for output_name, isolated_samples_list in output_data.items():
            self._output_data[output_name] = DatasetsBucketIsolatedSamplesListReader(
                isolated_samples=isolated_samples_list
            )

    @property
    def output_data(self):
        return self._output_data

    def generate_output_lists(self, ops: SnapshotOperations, samples_map):
        return samples_map
