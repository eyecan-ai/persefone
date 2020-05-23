from persefone.data.databases.h5 import H5SimpleDatabase


class DataReader(object):

    def __init__(self, database, columns):
        self.__database = database
        self.__columns = columns

        if isinstance(self.__columns, list):
            self.__columns = dict(zip(self.__columns, self.__columns))

        assert isinstance(self.__columns, dict), "Columns must be a dict!"

    @property
    def database(self):
        return self.__database

    @property
    def data(self):
        return self.__database.data

    @property
    def available_columns(self):
        return self.__columns

    @available_columns.setter
    def available_columns(self, columns):
        self.__columns = columns


class H5SimpleDataReader(DataReader):

    SPECIAL_COLUMNS_NAMES = ['_idx']
    REFERENCE_PREFIX = '@'
    PRIVATE_PREFIX = '_'

    def __init__(self, database, columns, enable_cache=False):
        """Reader specialization for H5Simple database tabular representation

        :param database: H5SimpleDatabase tabular representation as PandasDatabase
        :type database: PandasDatabase
        :param columns: list of columns to read
        :type columns: list
        """
        DataReader.__init__(self, database, columns)

        self.__filename_column = f'{self.PRIVATE_PREFIX}filename'
        assert self.__filename_column in self.data.columns, "Filename column is missing!"

        self.__cache_enabled = enable_cache
        self.__filenames_cache = {}

        self._purge_columns()

    def _purge_columns(self):
        new_columns = {}
        for column, new_name in self.available_columns.items():
            column_name, is_reference = self._purge_column_name(self.data.columns, column, force_prefix=True)
            if column_name is not None:
                new_columns[column_name] = new_name
        self.available_columns = new_columns

        for name, new_name in self.available_columns.items():
            if new_name.startswith(self.REFERENCE_PREFIX):
                self.available_columns[name] = new_name.replace(self.REFERENCE_PREFIX, '', 1)

    def _purge_column_name(self, columns, col_name, force_prefix=True):

        columns = list(columns) + self.SPECIAL_COLUMNS_NAMES
        is_reference = False
        retrieved_col_name = None
        if col_name.startswith(self.REFERENCE_PREFIX):
            is_reference = True
            if col_name in columns:
                retrieved_col_name = col_name
        else:
            if col_name in columns:
                retrieved_col_name = col_name
            else:
                if force_prefix:
                    return self._purge_column_name(columns, f'{self.REFERENCE_PREFIX}{col_name}', force_prefix=False)
        return retrieved_col_name, is_reference

    @property
    def cache_enabled(self):
        return self.__cache_enabled

    def _get_h5_resource(self, filename, reference):
        """Loads a h5py file resource by reference string

        :param filename: h5 filename
        :type filename: str
        :param reference: reference path
        :type reference: str
        :return: generic loaded data
        :rtype: np.ndarray
        """
        database = H5SimpleDatabase(filename=filename, readonly=True)
        data = None
        if self.cache_enabled:
            if filename not in self.__filenames_cache:
                self.__filenames_cache[filename] = database
                self.__filenames_cache[filename].open()
            data = self.__filenames_cache[filename][reference][...]
        else:
            with database:
                data = database[reference][...]
        return data

    def close(self):
        for filename, database in self.__filenames_cache.items():
            database.close()
        self.__filenames_cache = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        item = row.to_dict()
        item['_idx'] = row.name

        output_item = {}
        for col, new_name in self.available_columns.items():
            if col in item:
                if col.startswith(self.REFERENCE_PREFIX):
                    reference = item[col]
                    filename = item[self.__filename_column]
                    data = self._get_h5_resource(filename, reference)
                    output_item[new_name] = data
                else:
                    output_item[new_name] = item[col]

        return output_item
