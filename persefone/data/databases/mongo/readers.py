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


class MongoSamplesIterator(object):

    def __init__(self,
                 mongo_dataset: MongoDataset,
                 data_mapping: dict,
                 samples: List[MSample]
                 ):
        """ Generic iterator over MSample items

        :param mongo_dataset: target MongoDataset
        :type mongo_dataset: MongoDataset
        :param data_mapping: data mapping dict to filter MSample fields into output item
        :type data_mapping: dict
        :param samples: list of MSample to iterate
        :type samples: List[MSample]
        """

        self._mongo_dataset = mongo_dataset
        self._data_mapping = data_mapping
        self._samples = samples

    @property
    def mongo_dataset(self):
        return self._mongo_dataset

    @property
    def samples(self):
        return self._samples

    @property
    def data_mapping(self):
        return self._data_mapping

    def __len__(self):
        return len(self.samples)

    @classmethod
    def sample_to_item(cls, sample: MSample, data_mapping: dict, mongo_dataset: MongoDataset) -> dict:
        output_data = {}

        # Fetches plain data
        for k, v in sample.to_mongo().items():
            if k in data_mapping:
                output_data[data_mapping[k]] = v

        # Fetches MEtadata fields
        metadata = sample.metadata

        for k, v in metadata.items():
            extended_k = f'metadata.{k}'
            if extended_k in data_mapping:
                output_data[data_mapping[extended_k]] = v

        # Fetches Dataset fiels
        for k, v in sample.dataset.to_mongo().items():
            extended_k = f'dataset.{k}'
            if extended_k in data_mapping:
                output_data[data_mapping[extended_k]] = v

        # Fetches Items
        items = mongo_dataset.get_items(sample.sample_id)
        for item in items:
            item: MItem
            extended_name = f'items.{item.name}'
            if extended_name in data_mapping:
                data = None
                for resource in item.resources:
                    data = mongo_dataset.fetch_resource_to_numpyarray(resource)
                    break

                output_data[data_mapping[extended_name]] = data

        return output_data

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        sample: MSample = self.samples[idx]
        output_data = self.sample_to_item(sample, self.data_mapping, self.mongo_dataset)
        return output_data


class MongoDatasetReader(MongoSamplesIterator):

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

        super(MongoDatasetReader, self).__init__(mongo_dataset, data_mapping, [])
        self._query_dict = MongoQueryParser.parse_queries_list(queries)
        self._orders_bys = MongoQueryParser.parse_orders_list(orders)
        self._samples = list(self.mongo_dataset.get_samples(query_dict=self._query_dict, order_bys=self._orders_bys))

    @property
    def query_dict(self):
        return self._query_dict

    @property
    def orders(self):
        return self._orders_bys

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

    @classmethod
    def create_from_dict(cls, mongo_dataset: MongoDataset, cfg: dict) -> 'MongoDatasetReader':
        cfg = MongoDatasetReaderCFG.from_dict(cfg)
        cfg.validate()

        return MongoDatasetReader(
            mongo_dataset=mongo_dataset,
            data_mapping=cfg.params.data_mapping,
            queries=cfg.params.get('queries', []),
            orders=cfg.params.get('orders', [])
        )


class MongoMixedSamplesIterator(object):

    class TrampSample(object):

        def __init__(self, sample: MSample, data_mapping: dict, mongo_dataset: MongoDataset):
            """ A TrampSample is a MSample carring also its associated data_mapping and target dataset

            :param sample: plain MSample
            :type sample: MSample
            :param data_mapping: associated data_mapping dictionary
            :type data_mapping: dict
            :param mongo_dataset: target MongoDataset
            :type mongo_dataset: MongoDataset
            """
            self.sample = sample
            self.data_mapping = data_mapping
            self.mongo_dataset = mongo_dataset

    def __init__(self,
                 samples_dicts: List[TrampSample]
                 ):
        """ Creates a mixed samples iterator with MSample s coming from different MongoDataset

        :param samples_dicts: [description]
        :type samples_dicts: List[dict]
        """

        self._samples_dicts = samples_dicts

    def __len__(self):
        return len(self._samples_dicts)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        tramp_sample: self.TrampSample = self._samples_dicts[idx]

        output_data = MongoSamplesIterator.sample_to_item(
            tramp_sample.sample,
            tramp_sample.data_mapping,
            tramp_sample.mongo_dataset
        )
        return output_data
