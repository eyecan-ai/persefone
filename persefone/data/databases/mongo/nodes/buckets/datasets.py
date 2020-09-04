

import logging
from threading import Thread
from mongoengine.queryset.queryset import QuerySet
from persefone.data.databases.mongo.snapshots import SnapshotOperations
from schema import Optional, Schema
from persefone.utils.configurations import XConfiguration
from persefone.utils.mongo.queries import MongoQueryParser
from typing import Any, Dict, List, Sequence
from persefone.data.databases.mongo.nodes.nodes import MLink, MNode, NodesBucket
from mongoengine.errors import DoesNotExist
from persefone.data.databases.mongo.clients import MongoDatabaseClientCFG
from persefone.utils.bytes import DataCoding
import uuid


class DatasetsBucket(NodesBucket):

    DATASET_NAMSPACE_NAME = '$DATASETS'
    NODE_TYPE_DATASET = 'dataset'
    NODE_TYPE_SAMPLE = 'sample'
    NODE_TYPE_ITEM = 'item'
    LINK_TYPE_DATASET2SAMPLE = 'dataset_2_sample'
    LINK_TYPE_SAMPLE2ITEM = 'sample_2_item'
    DEFAULT_SAMPLE_ID_ZERO_PADDING = 7

    def __init__(self, client_cfg: MongoDatabaseClientCFG):
        """ Bucket for Datasets management

        :param client_cfg: database configuration object
        :type client_cfg: MongoDatabaseClientCFG
        """
        super(DatasetsBucket, self).__init__(client_cfg, self.DATASET_NAMSPACE_NAME)

    def get_datasets(self) -> Sequence[MNode]:
        """ Retrives a list of all datasets

        :return: list of datasets nodes
        :rtype: Sequence[MNode]
        """
        return self.get_namespace_node().outbound_nodes(link_type=self.LINK_TYPE_NAMESPACE2GENERIC)

    def new_dataset(self, dataset_name: str) -> MNode:
        """ Creates new dataset node by name

        :param dataset_name: dataset name
        :type dataset_name: str
        :raises NameError: raise Exectpion if name collision occurs
        :return: created MNode
        :rtype: MNode
        """

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

    def get_samples(self, dataset_name: str) -> Sequence[MNode]:
        """ Retrieves list of samples nodes of target database

        :param dataset_name: target database name
        :type dataset_name: str
        :return: list of retrived samples nodes
        :rtype: Sequence[MNode]
        """

        return self.get_samples_by_query(dataset_name=dataset_name)
        # dataset_node: MNode = self.get_dataset(dataset_name)
        # return dataset_node.outbound_nodes(link_type=self.LINK_TYPE_DATASET2SAMPLE)

    def count_samples(self, dataset_name: str) -> int:
        """ Counts samples nodes of target dataset

        :param dataset_name: target dataset name
        :type dataset_name: str
        :return: number of samples
        :rtype: int
        """
        return len(self.get_samples(dataset_name))

    def get_sample(self, dataset_name: str, sample_id: str) -> MNode:
        """  Retrives single sample node by id and target dataset

        :param dataset_name: target dataset name
        :type dataset_name: str
        :param sample_id: sample id
        :type sample_id: str
        :return: retrived sample node
        :rtype: MNode
        """
        return self.get_node_by_name(self.namespace / dataset_name / self._sample_id_name(sample_id))

    def get_samples_by_query(self, dataset_name: str, queries: list = None, orders_by: list = None) -> QuerySet:
        """ Retrieves sample nodes by query strings

        :param dataset_name: target dataset name
        :type dataset_name: str
        :param queries: plain queries strings (e.g. ['metadata.field_0 >= 33.3']) , defaults to None
        :type queries: list, optional
        :param orders_by: plain orders strngs (e.g. ['+metadata.field_X']), defaults to None
        :type orders_by: list, optional
        :return: list of retrieved MNode
        :rtype: MNode
        """

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

    def get_item(self, dataset_name: str, sample_id: str, item_name: str) -> MNode:
        """ Retrives item node by dataset/sample/name

        :param dataset_name: target dataset
        :type dataset_name: str
        :param sample_id: target sample id
        :type sample_id: str
        :param item_name: target item name
        :type item_name: str
        :return: retrived MNode
        :rtype: MNode
        """
        return self.get_node_by_name(self.namespace / dataset_name / self._sample_id_name(sample_id) / item_name)

    def get_items(self, dataset_name: str, sample_id: str) -> Sequence[MNode]:
        """ Retrives a list of all item nodes of target dataset/sample

        :param dataset_name: target dataset name
        :type dataset_name: str
        :param sample_id: target sample id
        :type sample_id: str
        :return: list of retrived item nodes
        :rtype: Sequence[MNode]
        """

        sample_node: MNode = self.get_sample(dataset_name, sample_id)
        return sample_node.outbound_nodes(link_type=self.LINK_TYPE_SAMPLE2ITEM)

    def delete_dataset(self, dataset_name: str, num_workers: int = 10):
        """ Recursive deletion of dataset

        :param dataset_name: target dataset name
        :type dataset_name: str
        :param num_workers: number of used thread
        :type num_workers: int
        """

        from queue import Queue
        garbage = Queue(maxsize=0)

        def destroyer(q: Queue):
            while not q.empty():
                q.get().delete()
                q.task_done()

        dataset_node = self.get_dataset(dataset_name)

        samples = dataset_node.outbound_nodes(link_type=self.LINK_TYPE_DATASET2SAMPLE)

        for sample_node in samples:
            items = sample_node.outbound_nodes(link_type=self.LINK_TYPE_SAMPLE2ITEM)
            for item_node in items:
                garbage.put(item_node)
                garbage.put(sample_node)
        garbage.put(dataset_node)

        workers = []
        for w in range(num_workers):
            worker = Thread(target=destroyer, args=(garbage,), daemon=True)
            worker.start()

        garbage.join()

    def _sample_id_name(self, sample_id: Any) -> str:
        """ Converts a sample id (Any) in a string

        :param sample_id: input sample id
        :type sample_id: Any
        :return: string representation
        :rtype: str
        """

        return str(sample_id)

    def new_sample(self, dataset_name: str, metadata: dict = None, sample_id: str = None) -> MNode:
        """ Creates new sample node

        :param dataset_name: target dataset
        :type dataset_name: str
        :param metadata: sample metadata to store, defaults to None
        :type metadata: dict, optional
        :param sample_id: new sample id, defaults to -1
        :type sample_id: str, optional
        :raises NameError: raise Exception if new sample id is used
        :return: created sample node
        :rtype: MNode
        """

        dataset_node: MNode = self.get_dataset(dataset_name)

        if sample_id is None:
            sample_id = str(uuid.uuid1())

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

    def new_item(self, dataset_name: str, sample_id: str, item_name: str, blob_data: bytes = None, blob_encoding: str = None) -> MNode:
        """ Creates Item from blob and encoding

        :param dataset_name: target dataset name
        :type dataset_name: str
        :param sample_id: target sample id
        :type sample_id: str
        :param item_name: new item name
        :type item_name: str
        :param blob_data: input blob data to store, defaults to None
        :type blob_data: bytes, optional
        :param blob_encoding: input blob encoding to store, defaults to None
        :type blob_encoding: str, optional
        :raises NameError: raise Exception if exits item with the same name
        :return: created item node
        :rtype: MNode
        """

        sample_node: MNode = self.get_sample(dataset_name, sample_id)

        try:
            self.get_item(dataset_name, sample_id, item_name)
            raise NameError(f"Item with same name '{item_name}' found")
        except DoesNotExist:

            item_node: MNode = self[self.namespace / dataset_name / self._sample_id_name(sample_id) / item_name]

            sample_node.link_to(item_node, link_type=self.LINK_TYPE_SAMPLE2ITEM)
            item_node.node_type = self.NODE_TYPE_ITEM
            item_node.put_data(blob_data, blob_encoding)
            return item_node

    @classmethod
    def remap_sample_with_items(cls, sample: MNode, data_mapping: dict) -> dict:
        """ Remaps a sample, with its items, into a plain dictionary

        :param sample: input sample
        :type sample: MNode
        :param data_mapping: data mapping dictionary to remap and filters fields
        :type data_mapping: dict
        :return: plain dictionary withre deduced/remapped info from sample and its items
        :rtype: dict
        """

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
        """ A DatasetsBucketReader deals with reading information from datasets bucket representation

        :param datasets_bucket: target bucket
        :type datasets_bucket: DatasetsBucket
        :param dataset_name: target dataset name
        :type dataset_name: str
        :param data_mapping: data mapping to apply, defaults to {}
        :type data_mapping: dict, optional
        :param queries: queries used to filter samples, defaults to []
        :type queries: List[str], optional
        :param orders: orders-queries to order samples, defaults to []
        :type orders: List[str], optional
        """

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
        """ It is a simple Samples iterator, with a commono datamapping schema

        :param samples: input samples list
        :type samples: List[MNode]
        :param data_mapping: data mapping to apply
        :type data_mapping: dict
        """

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
        """ An isolated samples is a plain sample MNode with its data mapping

        :param sample: input sample
        :type sample: MNode
        :param data_mapping: data mapping to apply
        :type data_mapping: dict
        """
        self.sample = sample
        self.data_mapping = data_mapping


class DatasetsBucketIsolatedSamplesListReader(object):

    def __init__(self, isolated_samples: List[DatasetsBucketIsolatedSample]):
        """ Isolated samples iterators deal with generic sample MNode despite its source dataset

        :param isolated_samples: list of DatasetsBucketIsolatedSample
        :type isolated_samples: List[DatasetsBucketIsolatedSample]
        """
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
    def output_data(self) -> Dict[str, DatasetsBucketIsolatedSamplesListReader]:
        """ Output data dictionary

        :return: dictionary of samples reader that can be used as data iterators
        :rtype: Dict[str, DatasetsBucketIsolatedSamplesListReader]
        """

        return self._output_data

    def generate_output_lists(self, ops: SnapshotOperations, samples_map):
        return samples_map
