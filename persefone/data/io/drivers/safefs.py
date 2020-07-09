from fs import open_fs
from persefone.utils.configurations import XConfiguration
from persefone.data.io.drivers.common import AbstractFileDriver
from schema import Schema, Optional
from pathlib import Path


class SafeFilesystemDriverCFG(XConfiguration):

    def __init__(self, filename):
        super(SafeFilesystemDriverCFG, self).__init__(filename=filename)
        self.set_schema(Schema({
            # NAME
            'base_folder': str,
        }))


class SafeFilesystemDriver(AbstractFileDriver):

    def __init__(self, cfg: SafeFilesystemDriverCFG):
        self.prefix = 'local'
        self._cfg = cfg

        self._base_folder = Path(self._cfg.params.base_folder)
        if not self._base_folder.exists():
            self._base_folder.mkdir(parents=True, exist_ok=False)

    @classmethod
    def driver_name(cls):
        return "safe_filesystem_driver"

    def _purge_uri(self, uri: str):
        if uri.startswith(self.full_prefix_qualifier):
            uri = uri.replace(self.full_prefix_qualifier, '')
            if uri.startswith('/'):
                raise LookupError(f"{self.__class__.__name__} cannot manage absolute URIs like: '{uri}'")
            return uri
        else:
            raise LookupError(f"{self.__class__.__name__} cannot manage URIs like: '{uri}'")

    def get(self, uri: str, mode: str = 'r'):
        puri = self._purge_uri(uri)
        puri_path = self._base_folder / Path(puri)
        puri_path.parent.mkdir(parents=True, exist_ok=True)
        return open(puri_path, mode)

    def delete(self, uri: str):
        puri = self._purge_uri(uri)
        puri_path: Path = self._base_folder / Path(puri)
        puri_path.unlink()
