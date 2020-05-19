from persefone.utils.configurations import XConfiguration
from persefone.data.databases.utils import H5SimpleDatabaseUtils
from persefone.data.databases.pandas import PandasDatabase
from persefone.utils.pyutils import get_arg
from schema import Schema, And, Or, Optional
from pathlib import Path


class SnapshotConfiguration(XConfiguration):

    def __init__(self, filename=None):
        XConfiguration.__init__(self, filename=filename)

        path_validator = str
        self.set_schema(Schema({
            # NAME
            'name': And(str),
            # RANDOM SEED
            Optional('random_seed'): And(int),
            # SOURCE LIST
            'sources': And(list, [path_validator]),
            # PIPELINE OBJECT
            Optional('queries'): [str],
            # PIPELINE OBJECT
            Optional('operations'): [{str: Or(int, float, list, str)}],
            # SPLITS DICTIONARY
            Optional('splits'): {str: Or(int, float)}
        }))


class DatabaseSnapshot(object):

    def __init__(self, filename):
        self.__cfgfile = Path(filename)
        self.__cfg = SnapshotConfiguration(filename=filename)
        self.__database = PandasDatabase()
        self.__reduced_database = PandasDatabase()
        self.__output_database_dictionary = {}
        self.__has_splits = False
        self.__name = ""
        self.__random_seed = 0
        if self.is_valid():
            self.__load_data()

    @property
    def name(self):
        return self.__name

    def __load_data(self):
        """Loads internal databases """

        if self.is_valid():

            self.__random_seed = get_arg(self.params, "random_seed", 0)
            self.__name = self.params.name
            self.__database = H5SimpleDatabaseUtils.h5files_to_pandas_database(self.sources, include_filenames=True)
            self.__reduced_database = self.__database.copy()

            queries = get_arg(self.params, "queries", [])
            self.__applies_queries(queries)

            operations = get_arg(self.params, "operations", [])
            self.__applies_operations(operations)

            splits = get_arg(self.params, "splits", None)
            self.__has_splits = splits != None
            self.__output_database_dictionary = self.__compute_splits(splits)

    @property
    def has_splits(self):
        """Checks if snapshot has multiple splits

        :return: TRUE for multiple splits
        :rtype: bool
        """
        return self.__has_splits

    def __compute_splits(self, split_dictionary=None):
        """Computes splits if any

        :param split_dictionary: a dictionary str/float representing splits percentages, defaults to None
        :type split_dictionary: dict, optional
        :return: dictionary str/PandasDatabase of splits
        :rtype: dict
        """
        if split_dictionary is not None:
            return self.__reduced_database.splits_as_dict(percentages_dictionary=split_dictionary, integrity=True)
        else:
            return {self.__name: self.reduced_database.copy()}

    def __applies_operations(self, operations):
        """ Applies operations list

        :param operations: list of operations
        :type operations: list
        """
        for operation in operations:
            for op_name, op_opts in operation.items():
                self.__apply_operation(
                    operation_name=op_name,
                    operation_opts=op_opts
                )

    def __applies_queries(self, queries):
        """Applies a list of queries

        :param queries: list of string pandas-compliant queries
        :type queries: list
        """
        self.__reduced_database = self.__reduced_database.query_list(queries)

    @property
    def output(self) -> dict:
        """Resulting dictionary of PandasDatabase

        :return: dictionary of PandasDatabase with names
        :rtype: dict
        """
        return self.__output_database_dictionary

    @property
    def database(self) -> PandasDatabase:
        """Get original PandasDatabase

        :return: original PandasDatabase
        :rtype: PandasDatabase
        """
        return self.__database

    @property
    def reduced_database(self) -> PandasDatabase:
        """Reduced PandasDatabase

        :return: progressive reduced PandasDatabase after queries and operations
        :rtype: PandasDatabase
        """
        return self.__reduced_database

    @property
    def params(self) -> dict:
        """Configuration parameters

        :return: Configuration parameters
        :rtype: dict
        """
        return self.__cfg.params

    def is_valid(self) -> bool:
        """Checks if configuration is valid

        :return: configuration validation
        :rtype: bool
        """
        return self.__cfg.is_valid()

    @property
    def sources(self) -> list:
        """Loaded sources

        :return: list of loaded sources paths
        :rtype: list
        """
        sources = []
        if self.is_valid():
            sources = self.params.sources
        return sources

    def __apply_operation(self, operation_name, operation_opts=None):
        """ Applies single operation on reduced database

        :param operation_name: operation string representation
        :type operation_name: str
        :param operation_opts: operation optional options, defaults to None
        :type operation_opts: object, optional
        """
        if operation_name == 'limit':
            size = operation_opts
            size = min(size, self.__reduced_database.size)
            self.__reduced_database = self.__reduced_database[:size]
        if operation_name == 'shuffle':
            count = operation_opts
            for i in range(count):
                self.__reduced_database = self.__reduced_database.shuffle(seed=self.__random_seed)
