
from persefone.utils.configurations import XConfiguration
from schema import Schema, Optional
from mongoengine import connect, disconnect, DEFAULT_CONNECTION_NAME
from enum import Enum
from persefone.data.databases.mongo.model import MTask, MTaskStatus
from persefone.data.databases.mongo.repositories import TasksRepository
from typing import Union, List


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


class DatabaseTaskManagerType(Enum):
    """ DatabaseTaskManager permission levels """

    TASK_CREATOR = 0
    TASK_WORKER = 1
    TASK_GOD = 2


class DatabaseTaskManager(object):

    def __init__(self, mongo_client: MongoDatabaseClient, manager_type: DatabaseTaskManagerType = DatabaseTaskManagerType.TASK_GOD):
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

    def get_tasks(self, status: Union[List[MTaskStatus], MTaskStatus, None] = None, last_first: bool = True, negate: bool = False) -> List[MTask]:
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

        allowed = [DatabaseTaskManagerType.TASK_CREATOR, DatabaseTaskManagerType.TASK_GOD]
        if self._manager_type not in allowed:
            raise PermissionError(f"New task not allowd for {self._manager_type}")

        source = self.__class__.__name__ + "@" + self._manager_type.name
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

        allowed = [DatabaseTaskManagerType.TASK_WORKER, DatabaseTaskManagerType.TASK_GOD]
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

        allowed = [DatabaseTaskManagerType.TASK_WORKER, DatabaseTaskManagerType.TASK_GOD]
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

        allowed = [DatabaseTaskManagerType.TASK_WORKER, DatabaseTaskManagerType.TASK_GOD]
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

        allowed = [DatabaseTaskManagerType.TASK_WORKER, DatabaseTaskManagerType.TASK_GOD]
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

        allowed = [DatabaseTaskManagerType.TASK_GOD]
        if self._manager_type not in allowed:
            raise PermissionError(f"Permanent task deletion not allowd for {self._manager_type}")

        task = TasksRepository.get_task_by_name(name)
        if task is not None:
            return TasksRepository.delete_task(task)

        return False
