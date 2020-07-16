from persefone.data.databases.mongo.model import MSample, MItem
from persefone.data.databases.mongo.clients import MongoDataset
from typing import List
from persefone.utils.mongo.queries import MongoQueryParser
from persefone.utils.configurations import XConfiguration
from schema import Schema, Optional


class MongoDatasetReaderCFG(XConfiguration):

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


class MongoDatasetReader(object):

    def __init__(self,
                 mongo_dataset: MongoDataset,
                 data_mapping: dict = {},
                 queries: List[str] = [],
                 orders: List[str] = []
                 ):
        """ Iterable reader over target MongoDataset

        :param mongo_dataset: target MongoDataset
        :type mongo_dataset: MongoDataset
        :param data_mapping: key mapping from metadata/items/fields to getitem, defaults to {}
        :type data_mapping: dict, optional
        :param queries: list of mongo-like queries, defaults to []
        :type queries: List[str], optional
        :param orders: list of mongo.like orders field, defaults to []
        :type orders: List[str], optional
        """

        self._mongo_dataset = mongo_dataset
        self._data_mapping = data_mapping

        self._query_dict = MongoQueryParser.parse_queries_list(queries)
        self._orders_bys = MongoQueryParser.parse_orders_list(orders)
        self._samples = list(self._mongo_dataset.get_samples(query_dict=self._query_dict, order_bys=self._orders_bys))

    @property
    def data_mapping(self):
        return self._data_mapping

    @property
    def query_dict(self):
        return self._query_dict

    @property
    def orders(self):
        return self._orders_bys

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        sample: MSample = self._samples[idx]
        output_data = {}

        # Fetches plain data
        for k, v in sample.to_mongo().items():
            if k in self._data_mapping:
                output_data[self._data_mapping[k]] = v

        # Fetches MEtadata
        metadata = sample.metadata

        for k, v in metadata.items():
            extended_k = f'metadata.{k}'
            if extended_k in self._data_mapping:
                output_data[self._data_mapping[extended_k]] = v

        # Fetches Items
        items = self._mongo_dataset.get_items(sample.sample_id)
        for item in items:
            item: MItem
            extended_name = f'items.{item.name}'
            if extended_name in self._data_mapping:
                data = None
                for resource in item.resources:
                    data = self._mongo_dataset.fetch_resource_to_numpyarray(resource)
                    break

                output_data[self._data_mapping[extended_name]] = data
        return output_data

    @classmethod
    def create_from_configuration_file(cls, mongo_dataset: MongoDataset, filename: str) -> 'MongoDatasetReader':
        cfg = MongoDatasetReaderCFG(filename=filename)
        cfg.validate()

        return MongoDatasetReader(
            mongo_dataset=mongo_dataset,
            data_mapping=cfg.params.data_mapping,
            queries=cfg.params.get('queries', []),
            orders=cfg.params.get('orders', [])
        )
