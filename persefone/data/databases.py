import pandas as pd
import numpy as np
from pathlib import Path
import time
import uuid
import yaml


class PandasDatabase(object):

    def __init__(self, data=None, attrs={}):
        # TODO: define if attrs is a copy, a deepcopy or the same instance?
        if attrs is None:
            attrs = {}
        if data is None:
            data = pd.DataFrame()
        assert type(data) == pd.DataFrame
        self.__data = data
        self.__data.attrs = attrs

    @property
    def size(self):
        return len(self.data)

    @property
    def attrs(self):
        return self.data.attrs

    @attrs.setter
    def attrs(self, d):
        self.data.attrs = d

    @property
    def data(self):
        return self.__data

    def set_index(self, name):
        self.__data = self.__data.set_index(name)

    def filter_rows_by_column_values(self, column_name, values, negate=False):
        """Filters dataset rows based on a target column and a set of values

        :param column_name: target column name
        :type column_name: str
        :param values: list of filtering values
        :type values: list
        :param negate: TRUE to obtain a negate version of filtering operation, defaults to False
        :type negate: bool, optional
        :return: filtered PandasDatabase
        :rtype: PandasDatabase
        """
        if not isinstance(values, list):
            values = [values]
        if not negate:
            dataset = PandasDatabase(data=self.data[self.data[column_name].isin(values)])
        else:
            dataset = PandasDatabase(data=self.data[~self.data[column_name].isin(values)])
        return dataset

    def split_by_column_values(self, column_name, list_of_values, check_sizes=False):
        """ Splits dataset in multiple datasets based on a target column and a list of values

        :param column_name: target column name
        :type column_name: str
        :param list_of_values: list of values to split
        :type list_of_values: list
        :param check_sizes: TRUE to check if parts sumup to original dataset; used when list_of_values map all values, defaults to False
        :type check_sizes: bool, optional
        :return: tuple of new PandasDataset
        :rtype: tuple
        """
        assert isinstance(list_of_values, list), f"{self.__class__.__name__}.split_by_column_values: 'list_of_values' must be a list!"
        splits = []
        sizes = []
        for lv in list_of_values:
            split = self.filter_rows_by_column_values(column_name=column_name, values=lv)
            splits.append(split)
            sizes.append(split.size)
        sizes = np.array(sizes)
        if check_sizes:
            assert sizes.sum() == self.size, f"{self.__class__.__name__}.split_by_column_values: splits size not sum up to total size!"
        return tuple(splits)

    def unique(self, column_name):
        """List of unique values by column

        :param column_name: target column name
        :type column_name: str
        :return: list of unique values found
        :rtype: list
        """
        if column_name == 'index':
            return np.unique(self.data.index)
        else:
            arr = self.data[column_name].to_numpy()
            return np.unique(arr)

    def count_grouped_values(self, column_name):
        """Counts grouped value by column

        :param column_name: target column name
        :type column_name: str
        :return: dictionary containing grouped counter. E.g. {<column_name>:{'label_0':256,'label_1':128,'label_3':14}}
        :rtype: dict
        """
        column = self.data[column_name].to_numpy()
        uniques = self.unique(column_name)
        counts_map = {}
        for v in uniques:
            counts_map[v] = np.count_nonzero(column == v)
        return {column_name: counts_map}

    def split(self, percentage=0.8):
        """Splits PandasDatabase in two PandasDatabase obejcts based on a percentage split value

        :param percentage: percentage split value, defaults to 0.8
        :type percentage: float, optional
        :return: pair of two PandasDatabase
        :rtype: [PandasDatabase,PandasDatabase]
        """
        percentage = np.clip(percentage, 0.0, 1.0)
        d0, d1 = np.split(self.data, [int(len(self.data) * percentage)])
        assert len(d0) + len(d1) == len(self.data), f"{self.__class__.__name__}.split: splits size not sum up to total size!"
        return PandasDatabase(data=d0, attrs=self.attrs), PandasDatabase(data=d1, attrs=self.attrs)

    def shuffle(self, seed=-1):
        """Produces a shuffled copy of the original PandasDatabase

        :param seed: controlled random seed , defaults to -1
        :type seed: int, optional
        :return: Shuffled copy of the original PandasDatabase
        :rtype: PandasDatabase
        """
        if seed >= 0:
            np.random.seed(seed)
        new_data = self.data.copy().sample(frac=1)
        if seed >= 0:
            np.random.seed(int(time.time()))
        return PandasDatabase(data=new_data, attrs=self.attrs)

    def is_valid_index(self, idx):
        """Checks if index is inside data range

        :param idx: target index
        :type idx: int
        :return: TRUE for valid index
        :rtype: bool
        """
        return 0 <= idx < self.size

    def has_duplicated_index(self):
        """Checks for duplicated indices

        :return: TRUE if at least one duplicated index was found
        :rtype: bool
        """
        return np.any(self.data.index.duplicated())

    def rebuild_indices(self):
        """Replace old indices with new random generated list

        :return: PandasDatabase copy with new indexing
        :rtype: PandasDatabase
        """
        index = self.data.copy().index.to_numpy()
        remap = {}
        for i in range(len(index)):
            remap[index[i]] = PandasDatabase.generate_random_id()
        newdata = self.data.rename(index=remap)
        return PandasDatabase(data=newdata, attrs=self.attrs)

    def check_intersection(self, other: 'PandasDatabase', subset='index'):
        """Checks if two datasets has intersection based on index or target columns

        :param other: target PandasDatabase
        :type other: PandasDatabase
        :param subset: columns to check intersection within, defaults to 'index'
        :type subset: str, optional
        :return: TRUE if intersection is not empty
        :rtype: bool
        """
        sumup = self + other

        if subset == 'index':
            return np.any(sumup.data.index.duplicated())
        else:
            s0 = set(self.unique(subset))
            s1 = set(other.unique(subset))
            # return np.any(sumup.data.duplicated(subset=subset))
            return len(s0.intersection(s1)) > 0

    def __getitem__(self, idx):
        """Fetches a sub PandasDatabase from the original onw

        :param idx: index or slice to pick
        :type idx: int, slice
        :return: PandasDatabase wrapping the picked items
        :rtype: PandasDatabase
        """
        if isinstance(idx, slice):
            if idx.start is not None and idx.stop is not None:
                assert self.is_valid_index(idx.start), f"{self.__class__.__name__}.__getitem__: slice start {idx.start} must be in [0,size)"
                assert self.is_valid_index(idx.stop), f"{self.__class__.__name__}.__getitem__: slice stop {idx.stop} must be in [0,size)"
            sub = self.data[idx]  # .reset_index(drop=True)
            return PandasDatabase(data=sub, attrs=self.attrs)
        else:
            assert self.is_valid_index(idx), f"{self.__class__.__name__}.__getitem__: index must be in [0,size)!"
            sub = self.data[idx:idx + 1]  # .reset_index(drop=True)
            return PandasDatabase(data=sub, attrs=self.attrs)

    def copy(self):
        """Copy PandasDatabase by pd.DataFrame copy

        :return: a copy of original PandasDatabase
        :rtype: PandasDatabase
        """
        return PandasDatabase(data=self.data.copy(), attrs=self.attrs)

    def __str__(self):
        """String representation of inner pd.DataFrame data

        :return: pd.DataFrame string representation
        :rtype: str
        """
        return self.data.to_string()

    def __add__(self, other: 'PandasDatabase'):
        """Sum operation between two PandasDatabase

        :param other: second PandasDatabase addendum
        :type other: PandasDatabase
        :return: a summation of two PandasDatabase
        :rtype: PandasDatabase
        """
        if self.size == 0:
            return other.copy()
        if other.size == 0:
            return self.copy()

        d0 = self.copy()
        d1 = other.copy()
        datas = pd.concat([d0.data, d1.data])
        return PandasDatabase(data=datas, attrs=self.attrs)  # TODO: is the copy of first Database attrs a correct behaviour?

    def __iadd__(self, other):
        """Sum operation between two PandasDatabase. It is a alias for PandasDatabase.__add__

        :param other: second PandasDatabase addendum
        :type other: PandasDatabase
        :return: a summation of two PandasDatabase
        :rtype: PandasDatabase
        """
        return self + other

    @classmethod
    def from_array(cls, array: list, attrs: dict = {}):
        return PandasDatabase(data=pd.DataFrame(array), attrs=attrs)

    @classmethod
    def generate_random_id(cls):
        return str(uuid.uuid1())  # TODO: attach a central logic to provide global UUID


class PandasDatabaseIO(object):
    """ PandasDatabase IO functions """

    @classmethod
    def build_metadata_filename_from_original_filename(cls, filename):
        """Generates metadata filename based on input filename

        :param filename: input filename
        :type filename: str
        :return: metadatada associated filename
        :rtype: str
        """
        return str(filename) + ".meta.yml"

    @classmethod
    def save_pickle(cls, database: PandasDatabase, filename):
        """Saves PICKLE binary version of PandasDatabase

        :param database: input PandasDatabase
        :type database: PandasDatabase
        :param filename: output filename
        :type filename: str, pathlib.Path
        """
        database.data.to_pickle(filename)
        PandasDatabaseIO.save_metadata(database, filename)

    @classmethod
    def load_pickle(cls, filename):
        """Loads PandasDatabase from PICKLE binary file

        :param filename: input filename
        :type filename: str, pathlib.Path
        :return: loaded PandasDatabase
        :rtype: PandasDatabase
        """
        data = pd.read_pickle(filename)
        data.attrs = {}
        metadata = PandasDatabaseIO.load_metadata(filename)
        return PandasDatabase(data=data, attrs=metadata)

    @classmethod
    def save_csv(cls, database: PandasDatabase, filename):
        """Saves CSV text version of PandasDatabase

        :param database: input PandasDatabase
        :type database: PandasDatabase
        :param filename: output filename
        :type filename: str, pathlib.Path
        """
        database.data.to_csv(filename)
        PandasDatabaseIO.save_metadata(database, filename)

    @classmethod
    def load_csv(cls, filename):
        """Loads PandasDatabase from CSV text file. Special columns
        like lists or complex object will be loaded as strings!!!

        :param filename: input filename
        :type filename: str, pathlib.Path
        :return: loaded PandasDatabase
        :rtype: PandasDatabase
        """

        metadata = PandasDatabaseIO.load_metadata(filename)
        index = None
        dtype = {}
        if 'index' in metadata:
            index = metadata['index']
            dtype[index] = str
        data = pd.read_csv(filename, dtype=dtype)
        if index is not None:
            data = data.set_index(index)
        data.attrs = {}
        return PandasDatabase(data=data, attrs=metadata)

    @classmethod
    def save_metadata(cls, database: PandasDatabase, filename):
        """Saves PandasDatabase metadata in separate YAML file

        :param database: target PandasDatabase
        :type database: PandasDatabase
        :param filename: output filename of original saved PandasDatabase, metadata filename will be computed from that
        :type filename: str, pathlib.Path
        """
        metafilename = PandasDatabaseIO.build_metadata_filename_from_original_filename(filename)
        with open(metafilename, 'w') as outfile:
            yaml.dump(database.data.attrs, outfile, default_flow_style=False)

    @classmethod
    def load_metadata(cls, filename):
        """Loads PandasDatabase metadata from separate YAML file

        :param filename: [description]
        :type filename: input filename of original saved PandasDatabase, metadata filename will be computed from that
        :return: metadata dictionary if any
        :rtype: dict
        """
        metafilename = PandasDatabaseIO.build_metadata_filename_from_original_filename(filename)
        if Path(metafilename).exists():
            with open(metafilename) as f:
                metadata = yaml.safe_load(f)
                return metadata
        return {}


class PandasDatabaseGenerator(object):
    """ Base class for PandasDatabase generator """


class PandasDatabaseGeneratorUnderscoreNotation(PandasDatabaseGenerator):
    """ PandasDatabase generator based on target folder with 'UNDERSCORE' notation """

    @classmethod
    def generate_from_folder(cls, folder, columns_lambdas={}):
        """Generates a PandasDatabase from folder with 'UNDERSCORE' notation files

        :param folder: target folder
        :type folder: str, pathlib.Path
        :param columns_lambdas: dict of lambads to manipulate corresponding filename.
                                e.g: if files like "00000_label.txt" needs to be loaded as numpy integer the corresponding
                                lambads will be: columns_lambdas={'label':lambda x: np.loadtxt(x).astype(int)}
                                , defaults to {}
        :type columns_lambdas: dict, optional
        :return: [description]
        :rtype: [type]
        """

        path = Path(folder)
        files = list(sorted(path.glob('*')))
        files = [x for x in files if not str(x.name).startswith('.')]
        items_map = {}
        for f in files:
            name = f.name
            stem = f.stem
            chunks = stem.split('_')
            if len(chunks) > 1:
                key = chunks[0]
                what = chunks[1]
                tag = chunks[2] if len(chunks) > 2 else ''
                # ext = f.suffix
                subkey = what
                if len(tag) > 0:
                    subkey += f".{tag}"

                if key not in items_map:
                    items_map[key] = {}

                obj = name
                if what in columns_lambdas:
                    obj = columns_lambdas[what](f)

                items_map[key][subkey] = obj

        items = []
        for k, v in items_map.items():
            item = {
                'id': k
            }
            item.update(v)
            items.append(item)
        data = pd.DataFrame(items)
        data = data.set_index('id')

        attrs = {
            'index': 'id'
        }
        return PandasDatabase(data=data, attrs=attrs)
