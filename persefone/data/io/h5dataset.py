import h5py
from persefone.utils.pyutils import get_arg


class H5Dataset(object):

    DATA_BRANCH_NAME = '_data'

    def __init__(self, filename, **kwargs):
        self.__filename = filename

        self.__readonly = get_arg(kwargs, "readonly", default=False)
        self.__compression = get_arg(kwargs, "compression", default='gzip')
        self.__compression_opts = get_arg(kwargs, "compression_opts", default=4)

        self.__handle = None

    def is_empty(self):
        if self.is_open():
            return len(self.handle.keys()) == 0

    def initialize(self):
        if self.is_open():
            self.handle.create_group(H5Dataset.DATA_BRANCH_NAME)

    @property
    def readonly(self):
        return self.__readonly

    @property
    def filename(self):
        return self.__filename

    @property
    def handle(self) -> h5py.File:
        return self.__handle

    def is_open(self):
        return self.handle is not None

    def open(self):
        if not self.is_open():
            self.__handle = h5py.File(self.filename, 'r' if self.readonly else 'a')
            if self.is_empty():
                self.initialize()

    def close(self):
        if self.is_open():
            self.__handle.close()
            self.__handle = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def _get_item_path(self, id):
        return f"{H5Dataset.DATA_BRANCH_NAME}/{id}"

    def get_item(self, id, force_create=True):
        item = None
        if self.is_open():
            key = self._get_item_path(id)

            if key not in self.handle:
                if force_create:
                    item = self.create_item(id)
            else:
                item = self.handle[key]
        return item

    def create_item(self, id):
        if self.is_open():
            return self.handle.create_group(self._get_item_path(id))

    def _get_data_path(self, id, name):
        return f"{self._get_item_path(id)}/{name}"

    def create_data(self, id, name, shape, dtype=None, maxshape=None):
        data = None
        if self.is_open():
            if self.readonly:
                raise Exception()
            else:
                item = self.get_item(id, force_create=True)
                data = item.create_dataset(
                    name,
                    shape=shape,
                    maxshape=maxshape,
                    dtype=dtype,
                    compression=self.__compression,
                    compression_opts=self.__compression_opts
                )
        return data

    def get_data(self, id, name):
        data = None
        if self.is_open():
            key = self._get_data_path(id, name)
            if key in self.handle:
                data = self.handle[key]
        return data
