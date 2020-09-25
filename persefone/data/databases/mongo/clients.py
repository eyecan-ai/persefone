
from persefone.utils.configurations import XConfiguration
from schema import Schema, Optional
from mongoengine import connect, disconnect, DEFAULT_CONNECTION_NAME
from pathlib import Path
import subprocess


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
            Optional('mock'): bool,
            Optional('timeout'): int
        }))


class MongoDumper(object):

    @classmethod
    def dump_to_files(cls, database_name: str, output_folder: str):
        output_folder = Path(output_folder)
        output_filename = output_folder / 'database_name.gz'
        cmd = f'mongodump --gzip --db={database_name} --archive={str(output_filename)}'
        subprocess.call(cmd.split(' '))


class MongoDatabaseClient(object):
    DROP_KEY_0 = 'dsajdiosacjpasokcasopkcxsapokdpokpo12k3po12k'
    DROP_KEY_1 = 'soidjsaoidjsaoijdxsaoxsakmx2132131231@@@!!!!'

    def __init__(self, cfg: MongoDatabaseClientCFG):
        """ Creates a Mongo Databsae client wrapper to manage connections and to provide
        a generic handle to each Object interacting with database

        :param cfg: MongoDatabaseClientCFG configuration
        :type cfg: MongoDatabaseClientCFG
        """

        self._cfg = cfg
        self._mock = False
        if 'mock' in self._cfg.params:
            self._mock = self._cfg.params.mock
        self._connection = None

    @property
    def cfg(self):
        return self._cfg

    @property
    def db_name(self):
        return self._cfg.params.db

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
        if self._connection is None:
            if not self._mock:
                self._connection = connect(
                    db=self._cfg.params.db,
                    alias=alias,
                    host=self._cfg.params.host,
                    port=self._cfg.params.port,
                    serverSelectionTimeoutMS=self._cfg.params.get('timeout', 1000)
                )
            else:
                self._connection = connect(
                    db=self._cfg.params.db,
                    alias=alias,
                    host='mongomock://localhost',
                    port=self._cfg.params.port,
                    maxIdleTimeMS=self._cfg.params.get('timeout', 1),
                    socketTimeoutMS=self._cfg.params.get('timeout', 1),
                    serverSelectionTimeoutMS=self._cfg.params.get('timeout', 1000)
                )
        return self

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
