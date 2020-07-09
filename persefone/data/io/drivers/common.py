from abc import ABC, abstractmethod
from typing import List
from pathlib import Path


class AbstractFileDriver(ABC):

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

    def uri_from_chunks(self, bucket: str, obj: str, filename: str) -> str:
        return self.full_prefix_qualifier + str(Path(bucket) / Path(obj) / Path(filename))
