
from persefone.utils.configurations import XConfiguration
from schema import Schema, Optional
from mongoengine import connect, disconnect, DEFAULT_CONNECTION_NAME
from enum import Enum
from persefone.data.databases.mongo.model import MTask, MTaskStatus
from persefone.data.databases.mongo.repositories import TasksRepository
from typing import Union


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

    def __init__(self, cfg: MongoDatabaseClientCFG, mock=False):
        self._cfg = cfg
        self._mock = False
        if 'mock' in self._cfg.params:
            self._mock = self._cfg.params.mock
        self._connection = None

    @property
    def connected(self):
        return self._connection is not None

    def connect(self, alias=DEFAULT_CONNECTION_NAME):
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

    def drop_database(self, db=None, key0=None, key1=None):
        if key0 == self.DROP_KEY_0 and key1 == self.DROP_KEY_1:
            if self._connection is not None:
                self._connection.drop_database(db if db is not None else self._cfg.params.db)

    def disconnect(self, alias=DEFAULT_CONNECTION_NAME):
        return disconnect(alias=alias)

    @classmethod
    def create_from_configuration_file(cls, filename):
        cfg = MongoDatabaseClientCFG(filename=filename)
        cfg.validate()
        return MongoDatabaseClient(cfg=cfg)


class DatabaseTaskManagerType(Enum):
    TASK_CREATOR = 0
    TASK_WORKER = 1
    TASK_GOD = 2


class DatabaseTaskManager(object):

    def __init__(self, mongo_client: MongoDatabaseClient, manager_type: DatabaseTaskManagerType = DatabaseTaskManagerType.TASK_GOD):
        self._mongo_client = mongo_client
        self._manager_type = manager_type

    def get_task(self, name: str) -> Union[MTask, None]:
        if not self._mongo_client.connected:
            self._mongo_client.connect()

        return TasksRepository.get_task_by_name(name)

    def get_tasks(self, status: Union[MTaskStatus, None] = None, last_first: bool = True):
        if not self._mongo_client.connected:
            self._mongo_client.connect()

        return list(TasksRepository.get_tasks(status=status, last_first=last_first))

    def new_task(self, name: str, input_payload={}):
        if not self._mongo_client.connected:
            self._mongo_client.connect()

        allowed = [DatabaseTaskManagerType.TASK_CREATOR, DatabaseTaskManagerType.TASK_GOD]
        if self._manager_type not in allowed:
            raise PermissionError(f"New task not allowd for {self._manager_type}")

        source = self.__class__.__name__ + "@" + self._manager_type.name
        return TasksRepository.new_task(name=name, source=source, input_payload=input_payload)

    def start_task(self, name: str, input_payload={}) -> MTask:
        if not self._mongo_client.connected:
            self._mongo_client.connect()

        allowed = [DatabaseTaskManagerType.TASK_WORKER, DatabaseTaskManagerType.TASK_GOD]
        if self._manager_type not in allowed:
            raise PermissionError(f"Start task not allowd for {self._manager_type}")

        task = TasksRepository.get_task_by_name(name)
        if task is not None:
            return TasksRepository.start_task(task)

        return None

    def work_on_task(self, name: str, work_payload={}) -> MTask:
        if not self._mongo_client.connected:
            self._mongo_client.connect()

        allowed = [DatabaseTaskManagerType.TASK_WORKER, DatabaseTaskManagerType.TASK_GOD]
        if self._manager_type not in allowed:
            raise PermissionError(f"Work on task not allowd for {self._manager_type}")

        task = TasksRepository.get_task_by_name(name)
        if task is not None:
            return TasksRepository.work_on_task(task, work_payload)

        return None

    def complete_task(self, name: str, output_payload) -> MTask:
        if not self._mongo_client.connected:
            self._mongo_client.connect()

        allowed = [DatabaseTaskManagerType.TASK_WORKER, DatabaseTaskManagerType.TASK_GOD]
        if self._manager_type not in allowed:
            raise PermissionError(f"Complete task not allowd for {self._manager_type}")

        task = TasksRepository.get_task_by_name(name)
        if task is not None:
            return TasksRepository.complete_task(task, output_payload)

        return None

    def cancel_task(self, name: str) -> MTask:
        if not self._mongo_client.connected:
            self._mongo_client.connect()

        allowed = [DatabaseTaskManagerType.TASK_WORKER, DatabaseTaskManagerType.TASK_GOD]
        if self._manager_type not in allowed:
            raise PermissionError(f"Cancel task not allowd for {self._manager_type}")

        task = TasksRepository.get_task_by_name(name)
        if task is not None:
            return TasksRepository.cancel_task(task)

        return None
