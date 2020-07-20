from persefone.utils.configurations import XConfiguration
from persefone.data.io.drivers.common import AbstractFileDriver
from persefone.data.databases.mongo.readers import MongoDatasetReader, MongoMixedSamplesIterator
from persefone.data.databases.mongo.clients import MongoDatabaseClient, MongoDataset
from schema import Schema, Optional
from typing import List
import logging
import random
import numpy as np


class MongoSnapshotCFG(XConfiguration):

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


class SnapshotOperations(object):

    @classmethod
    def shuffle(cls, d, seed=-1):
        if seed > 0:
            random.seed(seed)
        random.shuffle(d)
        return d

    @classmethod
    def split(cls, d, percentage):
        percentage = np.clip(percentage, 0.0, 1.0)
        plen = int(len(d) * percentage)
        d0 = d[:plen]
        d1 = d[plen:]
        assert len(d0) + len(d1) == len(d), f"splits size not sum up to total size! {len(d0)} + {len(d1)} != {len(d)}"
        return d0, d1


class MongoSnapshot(object):

    def __init__(self, mongo_client: MongoDatabaseClient, drivers: List[AbstractFileDriver], cfg: MongoSnapshotCFG):
        """ Creates a MongoSnapshot object used to mix multiple MongoDatasetReaders with split/shuffle/sum (spuffle)

        :param mongo_client: input MongoDatabaseClient
        :type mongo_client: MongoDatabaseClient
        :param drivers: list of AbstractFileDriver used insied MongoDatabase
        :type drivers: List[AbstractFileDriver]
        :param cfg: output configuration wrapped in a MongoSnapshotCFG
        :type cfg: MongoSnapshotCFG
        """

        self._cfg = cfg
        self._cfg.validate()
        self._mongo_client = mongo_client
        self._drivers = drivers

        self._readers_map = {}

        for dataset in cfg.params.datasets:
            reader_name = list(dataset.keys())[0]
            dataset_name = dataset[reader_name]['dataset']['name']

            # Mongo Dataset handle
            mongo_dataset = MongoDataset(
                self._mongo_client,
                dataset_name,
                '',
                drivers
            )

            # Reader
            dataset_reader = MongoDatasetReader.create_from_dict(
                mongo_dataset=mongo_dataset,
                cfg=dataset[reader_name]['reader']
            )

            self._readers_map[reader_name] = dataset_reader

        # Generates a TrampSample map befor spuffle
        self._samples_map = {}
        for reader_name, reader in self._readers_map.items():
            self._samples_map[reader_name] = []
            for sample in reader.samples:
                self._samples_map[reader_name].append(MongoMixedSamplesIterator.TrampSample(
                    sample=sample,
                    data_mapping=reader.data_mapping,
                    mongo_dataset=reader.mongo_dataset
                ))

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
        for output_name, samples_dicts in output_data.items():
            self._output_data[output_name] = MongoMixedSamplesIterator(samples_dicts=samples_dicts)

    @property
    def output_data(self):
        return self._output_data

    def generate_output_lists(self, ops: SnapshotOperations, samples_map):
        return samples_map
