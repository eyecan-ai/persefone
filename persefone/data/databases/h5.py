import h5py
from persefone.utils.pyutils import get_arg
from persefone.utils.filesystem import tree_from_underscore_notation_files, is_file_image, is_file_numpy_array, get_file_extension
from pathlib import Path
import numpy as np
import imageio
from io import BytesIO
import tqdm
from types import SimpleNamespace
import pandas as pd
import uuid


class DataCodec(object):

    IMAGE_CODECS = ['jpg', 'jpeg', 'png', 'tiff', 'bmp']

    @classmethod
    def encode_image_to_bytes(cls, image_array, encoding='jpg'):
        buff = BytesIO()
        imageio.imwrite(buff, image_array, format=encoding)
        return buff.getbuffer()

    @classmethod
    def decode_image_from_bytes(cls, buffer, encoding='jpg'):
        buff = BytesIO(buffer)
        return imageio.imread(buff.getbuffer(), format=encoding)

    @classmethod
    def encode_data_to_bytes(cls, array, encoding):
        if encoding in DataCodec.IMAGE_CODECS:
            return cls.encode_image_to_bytes(array, encoding=encoding)
        else:
            raise NotImplementedError(f"Unknown data encoder for: '{encoding}'")

    @classmethod
    def decode_data_from_bytes(cls, buffer, encoding):
        if encoding in DataCodec.IMAGE_CODECS:
            return cls.decode_image_from_bytes(buffer, encoding=encoding)
        else:
            raise NotImplementedError(f"Unknown data decoder for: '{encoding}'")


class H5DatasetCompressionMethods(object):

    NONE = SimpleNamespace(name=None, opts=None)
    GZIP = SimpleNamespace(name='gzip', opts=4)


class H5Database(object):

    def __init__(self, filename, **kwargs):
        """Generic H5Database

        :param filename: database filename
        :type filename: str
        """
        self.__filename = filename
        self.__readonly = get_arg(kwargs, "readonly", default=True)
        self.__handle = None

    @classmethod
    def purge_root_item(cls, root_item):
        """Purge a root item string

        :param root_item: input root item
        :type root_item: str
        :return: purge root item, remove invalid keys like empty strings
        :rtype: [type]
        """

        if len(root_item) == 0:
            root_item = '/'
        if root_item == '//':
            root_item = '/'
        if root_item[0] != '/':
            root_item = '/' + root_item
        if root_item[-1] != '/':
            root_item = root_item + '/'
        if root_item == len(root_item) * root_item[0]:
            root_item = '/'
        return root_item

    def is_empty(self):
        """Checks if H5Database is empty

        :return: TRUE if is empty
        :rtype: bool
        """
        if self.is_open():
            return len(self.handle.keys()) == 0

    def initialize(self):
        """Initialize database
        """
        if self.is_open():
            pass

    @classmethod
    def purge_key(cls, key):
        """Purges key if it is not compliat (i.e. if '/' is missing as first character)

        :param key: key to purge
        :type key: str
        :return: purged key
        :rtype: str
        """
        if not key.startswith('/'):
            return '/' + key
        return key

    @property
    def readonly(self):
        """Checks if database is in read only mode

        :return: TRUE for readonly mode
        :rtype: bool
        """
        return self.__readonly

    @property
    def filename(self):
        """Linked filename

        :return: filename
        :rtype: str
        """
        return self.__filename

    @property
    def handle(self) -> h5py.File:
        """Pointer to real h5py file

        :return: h5py database
        :rtype: h5py.File
        """
        return self.__handle

    def is_open(self):
        """TRUE if file is already open"""

        return self.handle is not None

    def open(self):
        """Opens related file"""
        if not self.is_open():
            self.__handle = h5py.File(self.filename, 'r' if self.readonly else 'a')
            if self.is_empty():
                self.initialize()

    def close(self):
        """Closes related file"""
        if self.is_open():
            self.__handle.close()
            self.__handle = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def get_group(self, key, force_create=True) -> h5py.Group:
        """Fetches a Group by key if any

        :param key: group key
        :type key: str
        :param force_create: TRUE to create Group if not present, defaults to True
        :type force_create: bool, optional
        :return: fetched Group if any
        :rtype: h5py.Group
        """
        group = None
        if self.is_open():
            if key not in self.handle:
                if force_create:
                    group = self.handle.create_group(key)
            else:
                group = self.handle[key]
        return group

    def get_data(self, group_key, name) -> h5py.Dataset:
        """Fetches Dataset if any

        :param group_key: group key
        :type group_key: str
        :param name: dataset name
        :type name: str
        :return: fetched Dataset if any
        :rtype: h5py.Dataset
        """
        data = None
        if self.is_open():
            group = self.get_group(key=group_key)
            if group is not None:
                if name in group:
                    data = group[name]
        else:
            pass  # TODO: logging when file is closed!
        return data

    def create_data(self, group_key, name, shape, dtype=None, maxshape=None, compression=H5DatasetCompressionMethods.NONE) -> h5py.Dataset:
        """Creates Dataset

        :param group_key: group key
        :type group_key: str
        :param name: dataset name
        :type name: str
        :param shape: dataset shape as np.darray shape
        :type shape: tuple
        :param dtype: data type, defaults to None
        :type dtype: numpy.dtype, optional
        :param maxshape: max allowed shape for resizable data, defaults to None
        :type maxshape: tuple, optional
        :param compression: compressione type
        :type compression: H5DatasetCompressionMethods type
        :raises Exception: Exception
        :return: created dataset
        :rtype: h5py.Dataset
        """
        data = None
        if self.is_open():
            group = self.get_group(group_key, force_create=True)
            data = group.create_dataset(
                name,
                shape=shape,
                maxshape=maxshape,
                dtype=dtype,
                compression=compression.name,
                compression_opts=compression.opts
            )
        else:
            pass  # TODO: logging when file is closed!
        return data

    def store_object(self, group_key, name, obj) -> h5py.Dataset:
        """Stores generic object as dataset

        :param group_key: group key
        :type group_key: str
        :param name: data name
        :type name: str
        :param obj: object to store. numpy array or byte array
        :type obj: list / np.ndarray
        :return: created h5py.Dataset
        :rtype: h5py.Dataset
        """
        data = None
        if self.is_open():
            group = self.get_group(group_key, force_create=True)
            group[name] = obj
            data = group[name]
        return data

    def store_encoded_data(self, group_key, name, array, encoding):
        """Stores encoded data array

        :param group_key: group key
        :type group_key: str
        :param name: dataset name
        :type name: str
        :param array: plain array data
        :type array: list or np.ndarray
        :param encoding: encoder name [e.g. 'jpg']
        :type encoding: str
        """
        data = self.store_object(
            group_key,
            name,
            DataCodec.encode_data_to_bytes(array, encoding=encoding)
        )
        if data is not None:
            data.attrs['_encoding'] = encoding

    def is_encoded_data(self, group_key, name) -> bool:
        """Checks if key/name is an encoded data database

        :param group_key: group key
        :type group_key: str
        :param name: database name
        :type name: str
        :return: TRUE if database contains encoded data (it uses attrs schema to understand it)
        :rtype: bool
        """
        data = self.get_data(group_key, name)
        if data is not None:
            if '_encoding' in data.attrs:
                return True
        return False

    def load_encoded_data(self, group_key, name):
        """Loads decoded data to array

        :param group_key: group key
        :type group_key: str
        :param name: dataset name
        :type name: str
        :raises AttributeError: Raises AttributeError if dataset doesn't have 'encodings' attributes
        :return: decoded data
        :rtype: object
        """
        data = self.get_data(group_key, name)
        if data is not None:
            if self.is_encoded_data(group_key, name):
                encoding = data.attrs['_encoding']
                decoded_data = DataCodec.decode_data_from_bytes(data[...], encoding=encoding)
                return decoded_data
            else:
                raise AttributeError(f'Encoding attribute is missing for {group_key}/{name}')
        return None


class H5DatabaseIO(object):

    @classmethod
    def generate_from_folder(cls, h5file, folder, **kwargs):
        """Generates a H5Database from an Underscore notation folder

        :param h5file: Output H5 filename
        :type h5file: str
        :param folder: Input data folder
        :type folder: str
        :return: generated H5Database
        :rtype: H5Database
        """
        database = H5Database(filename=h5file, readonly=False, **kwargs)

        compression = get_arg(kwargs, "compression", default=H5DatasetCompressionMethods.NONE.name)
        compression_opts = get_arg(kwargs, "compression_opts", default=H5DatasetCompressionMethods.NONE.opts)
        compression_method = SimpleNamespace(name=compression, opts=compression_opts)

        root_item = get_arg(kwargs, "root_item", '/')
        root_item = H5Database.purge_root_item(root_item)

        uuid_keys = get_arg(kwargs, "uuid_keys", False)

        with database:
            tree = tree_from_underscore_notation_files(folder)
            counter = 0
            for key, slots in tqdm.tqdm(tree.items()):
                if uuid_keys:
                    key = str(uuid.uuid1())
                key = f'{root_item}{key}'
                group = database.get_group(key)
                group.attrs['counter'] = counter
                group.attrs['oddity'] = counter % 2
                for item_name, filename in slots.items():
                    loaded_data = cls.load_data_from_file(filename)
                    if loaded_data is not None:
                        data = database.create_data(
                            key,
                            item_name,
                            loaded_data.shape,
                            loaded_data.dtype,
                            compression=compression_method
                        )
                        data[...] = loaded_data
                counter += 1

        return database

    @classmethod
    def load_data_from_file(cls, filename):
        """Loads np.darray data from file based on its content

        :param filename: input filename
        :type filename: str
        :return: loaded data array
        :rtype: np.darray
        """
        data = None
        filename = Path(filename)
        if is_file_image(filename):
            data = imageio.imread(filename)
        elif is_file_numpy_array(filename):
            extension = get_file_extension(filename)
            if extension in ['txt']:
                data = np.loadtxt(filename)
            elif extension in ['npy', 'npz']:
                data = np.load(filename)
        if data is not None:
            data = np.atleast_2d(data)
        return data


class H5SimpleDatabase(H5Database):

    DEFAULT_TABULAR_PRIVATE_TOKEN = '_'
    DEFAULT_TABULAR_REFERENCE_TOKEN = '@'
    DEFAULT_ROOT_ITEM = '/_items/'

    def __init__(self, filename, **kwargs):
        H5Database.__init__(self, filename=filename, **kwargs)
        self.__root_item = get_arg(kwargs, 'root_item', H5SimpleDatabase.DEFAULT_ROOT_ITEM)
        self.__root_item = H5SimpleDatabase.purge_root_item(self.__root_item)

    @property
    def root(self):
        """Gets root element of database. May be different from classical '/'

        :return: root h5py.Group
        :rtype: h5py.Group
        """
        if self.is_open():
            if self.__root_item == '/':
                return self.handle
            else:
                return self.handle[self.__root_item]
        return None

    @property
    def keys(self):
        """Retrieves plain list of keys

        :return: list of keys, empty if database is closed
        :rtype: list
        """
        if self.is_open():
            return list(self.root.keys())
        else:
            return []

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.is_open():
            if isinstance(idx, int):
                key = self.keys[idx]
                return self.root[key]
            elif isinstance(idx, str):
                return self.root[idx]
        return None

    def _remove_root_item_from_key(self, key):
        return key.replace(self.__root_item, '', 1)

    def generate_tabular_representation(self, include_filename=True):
        """Generates Tabular (PANDAS) Representation of current H5SimpleDatabase

        :param include_filename: TRUE to add special column pointing to original h5 filename, defaults to True
        :type include_filename: bool, optional
        :return: pandas.DataFrame containing tabular data
        :rtype: pandas.DataFrame
        """
        if self.is_open():
            rows = []
            for item_key in self.root:
                item = self[item_key]
                row = {}
                row.update(item.attrs)
                row['id'] = self._remove_root_item_from_key(item.name)
                row[H5SimpleDatabase.DEFAULT_TABULAR_REFERENCE_TOKEN] = item.name

                for data in list(item.values()):
                    simple_data_name = self._remove_root_item_from_key(data.name)
                    data_key = f'{H5SimpleDatabase.DEFAULT_TABULAR_REFERENCE_TOKEN}{Path(simple_data_name).stem}'
                    data_value = data.name
                    row[data_key] = data_value

                if include_filename:
                    row[f'{H5SimpleDatabase.DEFAULT_TABULAR_PRIVATE_TOKEN}filename'] = self.filename
                rows.append(row)

            frame = pd.DataFrame(rows).set_index('id')
            return frame
        return None
