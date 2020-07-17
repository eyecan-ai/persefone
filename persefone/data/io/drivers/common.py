from abc import ABC, abstractmethod
from typing import List
from pathlib import Path
import os
import stat

# TODO: add docs


class AbstractFileDriver(ABC):

    FLAG_NO_UNLINK = stat.UF_NOUNLINK
    FLAG_IMMUTABLE = stat.UF_IMMUTABLE

    def __init__(self):
        self._prefix = 'none'

    @classmethod
    def driver_name(cls):
        return 'abstract_file_driver'

    @property
    def prefix(self):
        return self._prefix

    @prefix.setter
    def prefix(self, prefix: str):
        self._prefix = prefix

    @property
    def full_prefix_qualifier(self):
        return f'{self.prefix}://'

    @abstractmethod
    def get(self, uri: str, mode: str = 'rb'):
        pass

    @abstractmethod
    def delete(self, uri: str):
        pass

    def uri_from_chunks(self, realm: str,  bucket: str, obj: str, filename: str) -> str:
        return self.full_prefix_qualifier + str(Path(realm) / Path(bucket) / Path(obj) / Path(filename))
