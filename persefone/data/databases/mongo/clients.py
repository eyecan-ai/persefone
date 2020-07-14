
from persefone.utils.configurations import XConfiguration
from schema import Schema, Optional
from mongoengine import connect, disconnect, DEFAULT_CONNECTION_NAME
from enum import Enum
from persefone.data.io.drivers.common import AbstractFileDriver
from persefone.data.databases.mongo.model import (
    MTask, MTaskStatus, MItem, MSample, MResource, MDataset
)
from persefone.data.databases.mongo.repositories import (
    TasksRepository, DatasetsRepository, DatasetCategoryRepository,
    SamplesRepository, ItemsRepository, ResourcesRepository
)
from typing import Union, List, Dict
from pathlib import Path
import numpy as np
from PIL import Image
import logging
import io


class MongoDatabaseClientCFG(XConfiguration):

    def __init__(self, filename=None):
        XConfiguration.__init__(self, filename=filename)
        self.set_schema(Schema({
            # NAME
            'host': str,
            'port': int,
            'user': str,
            'pass': str,
            'db': str,
            Optional('mock'): bool
        }))


class MongoDatabaseClient(object):
    DROP_KEY_0 = 'dsajdiosacjpasokcasopkcxsapokdpokpo12k3po12k'
    DROP_KEY_1 = 'soidjsaoidjsaoijdxsaoxsakmx2132131231@@@!!!!'

    def __init__(self, cfg: MongoDatabaseClientCFG):
        """ Creates a Mongo Databsae client wrapper to manage connections

        :param cfg: MongoDatabaseClientCFG configuration
        :type cfg: MongoDatabaseClientCFG
        """

        self._cfg = cfg
        self._mock = False
        if 'mock' in self._cfg.params:
            self._mock = self._cfg.params.mock
        self._connection = None

    @property
    def connected(self):
        return self._connection is not None

    def connect(self, alias: str = DEFAULT_CONNECTION_NAME):
        """ Connects to db

        :param alias: alias used for current connection, defaults to DEFAULT_CONNECTION_NAME
        :type alias: str, optional
        :return: database connection #TODO: what is this?
        :rtype: database connection
        """
        if not self._mock:
            self._connection = connect(
                db=self._cfg.params.db,
                alias=alias,
                host=self._cfg.params.host,
                port=self._cfg.params.port
            )
        else:
            self._connection = connect(
                db=self._cfg.params.db,
                alias=alias,
                host='mongomock://localhost',
                port=self._cfg.params.port
            )
        return self._connection

    def drop_database(self, db: str = None, key0: str = None, key1: str = None):
        """ Drops related database. Needs double security key to delete database, it is used to avoid unwanted actions

        :param db: databsase name, NONE for use configuration name, defaults to None
        :type db: str, optional
        :param key0: security key 0, defaults to None
        :type key0: str, optional
        :param key1: security key 1, defaults to None
        :type key1: str, optional
        """
        if key0 == self.DROP_KEY_0 and key1 == self.DROP_KEY_1:
            if self._connection is not None:
                self._connection.drop_database(db if db is not None else self._cfg.params.db)

    def disconnect(self, alias: str = DEFAULT_CONNECTION_NAME):
        """ Disconnect active connection

        :param alias: active alias, defaults to DEFAULT_CONNECTION_NAME
        :type alias: str, optional
        """

        disconnect(alias=alias)

    @classmethod
    def create_from_configuration_file(cls, filename: str):
        """ Helper function to create a MongoDatabaseClient from configuration filename

        :param filename: configuration filename
        :type filename: str
        :return: created MongoDatabaseClient
        :rtype: MongoDatabaseClient
        """

        cfg = MongoDatabaseClientCFG(filename=filename)
        cfg.validate()
        return MongoDatabaseClient(cfg=cfg)


class MongoDatabaseTaskManagerType(Enum):
    """ DatabaseTaskManager permission levels """

    TASK_CREATOR = 0
    TASK_WORKER = 1
    TASK_GOD = 2


class MongoDatabaseTaskManager(object):

    def __init__(self, mongo_client: MongoDatabaseClient, manager_type: MongoDatabaseTaskManagerType = MongoDatabaseTaskManagerType.TASK_GOD):
        """ Creates a DatabaseTaskManager as interface to tasks management

        :param mongo_client: MongoDatabaseClient used for database connection
        :type mongo_client: MongoDatabaseClient
        :param manager_type: permission levels as DatabaseTaskManagerType, defaults to DatabaseTaskManagerType.TASK_GOD
        :type manager_type: DatabaseTaskManagerType, optional
        """

        self._mongo_client = mongo_client
        self._manager_type = manager_type

    def get_task(self, name: str) -> Union[MTask, None]:
        """ Retrieves single task by name

        :param name: task name
        :type name: str
        :return: target MTask or None if name not found
        :rtype: Union[MTask, None]
        """

        if not self._mongo_client.connected:
            self._mongo_client.connect()

        return TasksRepository.get_task_by_name(name)

    def get_tasks(self,
                  status: Union[List[MTaskStatus], MTaskStatus, None] = None,
                  last_first: bool = True, negate: bool = False) -> List[MTask]:
        """ Retrieves MTask list based on status, or whole list

        :param status: filtering MTaskStatus as single value or list, defaults to None
        :type status: Union[List[MTaskStatus], MTaskStatus, None], optional
        :param last_first: TRUE to obtain last tasks first, defaults to True
        :type last_first: bool, optional
        :param negate: TRUE to negate query results, defaults to False
        :type negate: bool, optional
        :return: list of retrieved MTask
        :rtype: List[MTask]
        """

        if not self._mongo_client.connected:
            self._mongo_client.connect()

        return list(TasksRepository.get_tasks(status=status, last_first=last_first, negate=negate))

    def new_task(self, name: str, input_payload={}) -> Union[MTask, None]:
        """ Creates new task with target name. Can caouse collisions

        :param name: choosen name
        :type name: str
        :param input_payload: initial metadata, defaults to {}
        :type input_payload: dict, optional
        :raises PermissionError: controls permission level
        :return: retrieved MTask or None if a task with the same name already exists
        :rtype: Union[MTask, None]
        """

        if not self._mongo_client.connected:
            self._mongo_client.connect()

        allowed = [MongoDatabaseTaskManagerType.TASK_CREATOR, MongoDatabaseTaskManagerType.TASK_GOD]
        if self._manager_type not in allowed:
            raise PermissionError(f"New task not allowd for {self._manager_type}")

        source = MongoDatabaseTaskManager.__name__ + "@" + self._manager_type.name
        return TasksRepository.new_task(name=name, source=source, input_payload=input_payload)

    def start_task(self, name: str) -> Union[MTask, None]:
        """ Starts a task

        :param name: target task name
        :type name: str
        :raises PermissionError: controls permission level
        :return: target task if start is ok, otherwise None
        :rtype: Union[MTask, None]
        """

        if not self._mongo_client.connected:
            self._mongo_client.connect()

        allowed = [MongoDatabaseTaskManagerType.TASK_WORKER, MongoDatabaseTaskManagerType.TASK_GOD]
        if self._manager_type not in allowed:
            raise PermissionError(f"Start task not allowd for {self._manager_type}")

        task = TasksRepository.get_task_by_name(name)
        if task is not None:
            return TasksRepository.start_task(task)

        return None

    def work_on_task(self, name: str, work_payload={}) -> Union[MTask, None]:
        """ Works on task updating metadata

        :param name: target task name
        :type name: str
        :param work_payload: working metadata, defaults to {}
        :type work_payload: dict, optional
        :raises PermissionError: controls permission level
        :return: target task if work is ok, otherwise None
        :rtype: Union[MTask, None]
        """

        if not self._mongo_client.connected:
            self._mongo_client.connect()

        allowed = [MongoDatabaseTaskManagerType.TASK_WORKER, MongoDatabaseTaskManagerType.TASK_GOD]
        if self._manager_type not in allowed:
            raise PermissionError(f"Work on task not allowd for {self._manager_type}")

        task = TasksRepository.get_task_by_name(name)
        if task is not None:
            return TasksRepository.work_on_task(task, work_payload)

        return None

    def complete_task(self, name: str, output_payload={}) -> Union[MTask, None]:
        """ Completes target task with output metadata

        :param name: target task name
        :type name: str
        :param output_payload: completion metadata, defaults to {}
        :type output_payload: dict, optional
        :raises PermissionError: controls permission level
        :return: target task if complete is ok, otherwise None
        :rtype: Union[MTask, None]
        """

        if not self._mongo_client.connected:
            self._mongo_client.connect()

        allowed = [MongoDatabaseTaskManagerType.TASK_WORKER, MongoDatabaseTaskManagerType.TASK_GOD]
        if self._manager_type not in allowed:
            raise PermissionError(f"Complete task not allowd for {self._manager_type}")

        task = TasksRepository.get_task_by_name(name)
        if task is not None:
            return TasksRepository.complete_task(task, output_payload)

        return None

    def cancel_task(self, name: str) -> Union[MTask, None]:
        """ Cancels task

        :param name: target task name
        :type name: str
        :raises PermissionError: control permission level
        :return: target task if cancel is ok, otherwise None
        :rtype: Union[MTask, None]
        """

        if not self._mongo_client.connected:
            self._mongo_client.connect()

        allowed = [MongoDatabaseTaskManagerType.TASK_WORKER, MongoDatabaseTaskManagerType.TASK_GOD]
        if self._manager_type not in allowed:
            raise PermissionError(f"Cancel task not allowd for {self._manager_type}")

        task = TasksRepository.get_task_by_name(name)
        if task is not None:
            return TasksRepository.cancel_task(task)

        return None

    def remove_task(self, name: str) -> bool:
        """ Permanently remove a target task

        :param name: target task name
        :type name: str
        :raises PermissionError:  control permission level
        :return: TRUE if task was removed
        :rtype: bool
        """

        if not self._mongo_client.connected:
            self._mongo_client.connect()

        allowed = [MongoDatabaseTaskManagerType.TASK_GOD]
        if self._manager_type not in allowed:
            raise PermissionError(f"Permanent task deletion not allowd for {self._manager_type}")

        task = TasksRepository.get_task_by_name(name)
        if task is not None:
            return TasksRepository.delete_task(task)

        return False


class MongoDatasetsManager(object):

    def __init__(self, mongo_client: MongoDatabaseClient):
        """ MongoDatasetsManager interface to datasets management

        :param mongo_client: MongoDatabaseClient used for database connection
        :type mongo_client: MongoDatabaseClient
        :type manager_type: DatabaseTaskManagerType, optional
        """

        self._mongo_client = mongo_client

    def get_datasets(self, dataset_name: str, drivers: Union[Dict[str, AbstractFileDriver], List[AbstractFileDriver]]):
        """ Retrievers available datasets

        :param dataset_name: dataset name query string (can be empty for all)
        :type dataset_name: str
        :param drivers: list or dict of AbstractFileDriver
        :type drivers: Union[Dict[str, AbstractFileDriver], List[AbstractFileDriver]]
        :return: list of MongoDataset
        :rtype: List[MDataset]
        """
        if not self._mongo_client.connected:
            self._mongo_client.connect()

        raw_datasets = list(DatasetsRepository.get_datasets(dataset_name=dataset_name))
        mongo_datasets = []
        for raw_dataset in raw_datasets:
            mongo_dataset = MongoDataset(
                mongo_client=self._mongo_client,
                dataset_name=raw_dataset.name,
                dataset_category=raw_dataset.category.name,
                drivers=drivers
            )
            mongo_datasets.append(mongo_dataset)
        return mongo_datasets

    def get_dataset(self, dataset_name: str, drivers: Union[Dict[str, AbstractFileDriver], List[AbstractFileDriver]]):
        """ Retrives single dataset by name

        :param dataset_name: dataset name
        :type dataset_name: str
        :param drivers: list or dict of AbstractFileDriver
        :type drivers: Union[Dict[str, AbstractFileDriver], List[AbstractFileDriver]]
        :return: single MongoDataset
        :rtype: MongoDataset
        """

        if not self._mongo_client.connected:
            self._mongo_client.connect()

        raw_dataset = DatasetsRepository.get_dataset(dataset_name=dataset_name)
        if raw_dataset is not None:
            mongo_dataset = MongoDataset(
                mongo_client=self._mongo_client,
                dataset_name=raw_dataset.name,
                dataset_category=raw_dataset.category.name,
                drivers=drivers
            )
            return mongo_dataset

    def create_dataset(self, dataset_name: str, dataset_category: str, drivers: Union[Dict[str, AbstractFileDriver], List[AbstractFileDriver]]):
        """ Creates MongoDataset

        :param dataset_name: dataset name
        :type dataset_name: str
        :param dataset_category: dataset category
        :type dataset_category: str
        :param drivers: list or dict of AbstractFileDriver
        :type drivers: Union[Dict[str, AbstractFileDriver], List[AbstractFileDriver]]
        :return: created MongoDataset
        :rtype: MongoDataset
        """

        if not self._mongo_client.connected:
            self._mongo_client.connect()

        try:
            mongo_dataset = MongoDataset(
                mongo_client=self._mongo_client,
                dataset_name=dataset_name,
                dataset_category=dataset_category,
                drivers=drivers,
                error_if_exists=True
            )
            return mongo_dataset
        except Exception as e:
            logging.error(e)
            return None


class MongoDataset(object):

    def __init__(self,
                 mongo_client: MongoDatabaseClient,
                 dataset_name: str,
                 dataset_category: str,
                 drivers: Union[Dict[str, AbstractFileDriver], List[AbstractFileDriver]] = {},
                 create_if_none: bool = True,
                 error_if_exists: bool = False):
        """ Database Dataset client for read/write operation

        :param mongo_client: mongo db client object
        :type mongo_client: MongoDatabaseClient
        :param dataset_name: target dataset name
        :type dataset_name: str
        :param dataset_category: target dataset category
        :type dataset_category: str
        :param create_if_none: TRUE to create if not found
        :type create_if_none: bool
        :param error_if_exists: TRUE to raise error on dataset with same name
        :type error_if_exists: bool
        :param drivers: dictionary with file drivers or list of AbstractFileDriver, defaults to {}
        :type drivers: Union[Dict[str, AbstractFileDriver], List[AbstractFileDriver]], optional
        """

        self._client = mongo_client
        if isinstance(drivers, dict):
            self._drivers = drivers
        elif isinstance(drivers, list):
            self._drivers = {}
            for driver in drivers:
                driver: AbstractFileDriver
                self._drivers[driver.driver_name()] = driver
        else:
            raise NotImplementedError("drivers is neither a dict nor a list of AbstractFileDriver")

        self._client.connect()

        self._dataset = DatasetsRepository.get_dataset(dataset_name)
        if error_if_exists and self._dataset is not None:
            raise Exception("Dataset with the same name found!")

        if self._dataset is None and create_if_none:
            self._dataset = DatasetsRepository.new_dataset(dataset_name, dataset_category)
            assert self._dataset is not None, "Something goes wrong creating Dataset"

    @property
    def dataset(self):
        return self._dataset

    def delete(self, security_name: str) -> bool:
        """ Deletes dataset with secury name check

        :param security_name: name of the dataset, used for security check
        :type security_name: str
        :return: TRUE if deletion complete
        :rtype: bool
        """

        if security_name == self._dataset.name:
            samples = self.get_samples()
            for sample in samples:
                items = self.get_items(sample.sample_id)
                for item in items:
                    for resource in item.resources:
                        resource: MResource

                        if resource.driver in self._drivers:
                            driver = self._drivers[resource.driver]
                            driver.delete(resource.uri)
                        else:
                            raise ModuleNotFoundError(f"No avaibale driver [{resource.driver}] to delete resource!")

                        resource.delete()
                    item.delete()
                sample.delete()
            self._dataset.delete()
            return True
        return False

    def get_sample(self, sample_idx: int) -> Union[MSample, None]:
        """ Retrieves single sample by idx

        :param sample_idx: sample idx
        :type sample_idx: int
        :return: MSample object or None
        :rtype: Union[MSample, None]
        """
        return SamplesRepository.get_sample_by_idx(self._dataset, sample_idx)

    def get_samples(self) -> List[MSample]:
        """ Retrives all samples of current dataset

        :return: list of MSample
        :rtype: List[MSample]
        """
        return list(SamplesRepository.get_samples(dataset=self._dataset))

    def count_samples(self) -> int:
        """ Counts samples

        :return: number of samples
        :rtype: int
        """

        return SamplesRepository.count_samples(dataset=self._dataset)

    def add_sample(self, metadata: dict = {}) -> Union[MSample, None]:
        """ Creates new sample. If a idx collision occurs, None is returned

        :param metadata: sample metadata, defaults to {}
        :type metadata: dict, optional
        :return: MSample object or None
        :rtype: Union[MSample, None]
        """

        return SamplesRepository.new_sample(self._dataset, -1, metadata)

    def get_item(self, sample_idx: int, item_name: str) -> Union[MItem, None]:
        """ Retrieves single sample item

        :param sample_idx: sample idx
        :type sample_idx: int
        :param item_name: item name
        :type item_name: str
        :return: an MItem or None
        :rtype: Union[MItem, None]
        """

        sample = SamplesRepository.get_sample_by_idx(self._dataset, sample_idx)
        if sample is not None:
            return ItemsRepository.get_item_by_name(sample, item_name)
        else:
            return None

    def get_items(self, sample_idx: int) -> List[MItem]:
        """ Retrieves all items of given sample

        :param sample_idx: sample idx
        :type sample_idx: int
        :return: list of MItem
        :rtype: List[MItem]
        """

        sample = SamplesRepository.get_sample_by_idx(self._dataset, sample_idx)
        if sample is not None:
            return ItemsRepository.get_items(sample=sample)
        else:
            return None

    def add_item(self, sample_idx: int, item_name: str) -> Union[MItem, None]:
        """ Creates new item associated with sample

        :param sample_idx: sample idx
        :type sample_idx: int
        :param item_name: item name
        :type item_name: str
        :return: an MItem or None if errors occur
        :rtype: Union[MItem, None]
        """

        sample = SamplesRepository.get_sample_by_idx(self._dataset, sample_idx)
        if sample is not None:
            return ItemsRepository.new_item(sample, item_name)
        return None

    def push_resource(self,
                      sample_idx: int,
                      item_name: str,
                      resource_name: str,
                      filename_to_copy: str,
                      driver_name: str) -> Union[MResource, None]:
        """ Creates a resource pushing an external file within

        :param sample_idx: sample idx
        :type sample_idx: int
        :param item_name: item name
        :type item_name: str
        :param resource_name: resource name to store
        :type resource_name: str
        :param filename_to_copy: source filename
        :type filename_to_copy: str
        :param driver_name: driver name used to create the resource
        :type driver_name: str
        :raises KeyError: If driver does not exist in dataset available drivers
        :return: TRUE if creatioin was ok
        :rtype: bool
        """

        with open(filename_to_copy, 'rb') as fin:
            return self.push_resource_from_blob(
                sample_idx, item_name, resource_name, fin.read(), Path(filename_to_copy).suffix, driver_name
            )

    def push_resource_from_blob(self,
                                sample_idx: int,
                                item_name: str,
                                resource_name: str,
                                blob: bytes,
                                extension: str,
                                driver_name: str) -> Union[MResource, None]:
        """ Creates new resource pushing a byte array within

        :param sample_idx: target sample idx
        :type sample_idx: int
        :param item_name: target item name
        :type item_name: str
        :param resource_name: resource name to store
        :type resource_name: str
        :param blob: source bytes array
        :type blob: bytes
        :param extension: extension for resource
        :type extension: str
        :param driver_name: name of driver used to store data
        :type driver_name: str
        :raises KeyError: If driver does not exist in dataset available drivers
        :return: MResource istance or None if errors occur
        :rtype: Union[MResource, None]
        """

        if driver_name not in self._drivers:
            raise KeyError(f"Driver '{driver_name}' not found!")

        driver: AbstractFileDriver = self._drivers[driver_name]

        item = self.get_item(sample_idx, item_name)

        if item is not None:
            if '.' not in extension:
                extension = f'.{extension}'
            uri = driver.uri_from_chunks(self._dataset.name, str(sample_idx), item_name + f'{extension}')
            resource = ItemsRepository.create_item_resource(item, resource_name, driver_name, uri)
            if resource is not None:
                with driver.get(uri, 'wb') as fout:
                    fout.write(blob)
            return resource
        return None

    def fetch_resource_to_blob(self,
                               resource: MResource) -> bytes:
        """ Fetches a resource into bytes array 

        :param resource: target resource
        :type resource: MResource
        :raises KeyError: If driver does not exist in dataset available drivers
        :return: bytes array
        :rtype: bytes
        """

        driver_name = resource.driver
        if driver_name not in self._drivers:
            raise KeyError(f"Driver '{driver_name}' not found!")

        driver: AbstractFileDriver = self._drivers[driver_name]

        blob = bytes()
        with driver.get(resource.uri, 'rb') as fin:
            blob = fin.read()

        return blob

    def fetch_resource_to_numpyarray(self, resource: MResource) -> np.ndarray:
        """ Fetches a resource into numpy array

        :param resource: target resoruce
        :type resource: MResource
        :raises NotmplementedError: If driver does not exist in dataset available drivers
        :return: numpy array
        :rtype: np.ndarray
        """

        blob = self.fetch_resource_to_blob(resource)
        # print(resource.uri, len(blob))
        fin = io.BytesIO(blob)

        try:
            data = np.array(Image.open(fin))
        except Exception as e:
            logging.debug(e)
            try:
                data = np.loadtxt(fin)
            except Exception as e:
                logging.debug(e)
                try:
                    data = np.load(fin)
                except Exception as e:
                    logging.debug(e)
                    raise NotImplementedError("Blob data is not interpretable!")

        return data


class MongoDatasetReader(object):

    def __init__(self,
                 mongo_dataset: MongoDataset,
                 data_mapping: dict = {},
                 preferred_driver: str = None):
        """ Reader wrapper for a MongoDataset

        :param mongo_dataset: target MongoDataset
        :type mongo_dataset: MongoDataset
        :param data_mapping: key mapping to retrieves metadata and items in final output, defaults to {}
        :type data_mapping: dict, optional
        :param preferred_driver: the preferred driver used during data fetches, defaults to None
        :type preferred_driver: str, optional
        """

        self._mongo_dataset = mongo_dataset
        self._data_mapping = data_mapping
        self._preferred_driver = preferred_driver

    def __len__(self):
        return self._mongo_dataset.count_samples()

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        sample: MSample = self._mongo_dataset.get_sample(idx)

        metadata = sample.metadata
        items = self._mongo_dataset.get_items(sample.sample_id)

        output_data = {}
        for k, v in metadata.items():
            if k in self._data_mapping:
                output_data[self._data_mapping[k]] = v

        for item in items:
            item: MItem
            if item.name in self._data_mapping:

                data = None
                for resource in item.resources:
                    data = self._mongo_dataset.fetch_resource_to_numpyarray(resource)
                    break

                output_data[self._data_mapping[item.name]] = data

        return output_data
