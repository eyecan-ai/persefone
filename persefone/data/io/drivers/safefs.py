from fs import open_fs
from persefone.utils.configurations import XConfiguration
from persefone.data.io.drivers.common import AbstractFileDriver
from schema import Schema, Optional
from pathlib import Path
import os


class SafeFilesystemDriverCFG(XConfiguration):

    def __init__(self, filename):
        super(SafeFilesystemDriverCFG, self).__init__(filename=filename)
        self.set_schema(Schema({
            # NAME
            'base_folder': str,
        }))


class SafeFilesystemDriver(AbstractFileDriver):

    def __init__(self, cfg: SafeFilesystemDriverCFG):
        """ Safe filesystem wrapper. It uses a target folder as root

        :param cfg: SafeFilesystemDriverCFG configuration
        :type cfg: SafeFilesystemDriverCFG
        """

        self.prefix = 'local'
        self._cfg = cfg

        self._base_folder = Path(self._cfg.params.base_folder)
        if not self._base_folder.exists():
            self._base_folder.mkdir(parents=True, exist_ok=False)

    @classmethod
    def driver_name(cls):
        return "safe_filesystem_driver"

    def _purge_uri(self, uri: str) -> str:
        """ Purges source uri removing driver qualifier and checking permissions on path

        :param uri: source uri
        :type uri: str
        :raises PermissionError: No absolute paths allowed
        :raises PermissionError: No Relative Path allowd
        :raises LookupError: unrecognized path
        :return: purged uri
        :rtype: str
        """

        if uri.startswith(self.full_prefix_qualifier):
            uri = uri.replace(self.full_prefix_qualifier, '')
            if uri.startswith('/'):
                raise PermissionError(f"{self.__class__.__name__} cannot manage absolute URIs like: '{uri}'")
            if '..' in uri:
                raise PermissionError(f"{self.__class__.__name__} cannot manage relative URIs like: '{uri}'")
            return uri
        else:
            raise LookupError(f"{self.__class__.__name__} cannot manage URIs like: '{uri}'")

    def get(self, uri: str, mode: str = 'r'):
        """ Gets target uri resource

        :param uri: resource uri
        :type uri: str
        :param mode: mode in ['r','w','rb','wb' ...], defaults to 'r'
        :type mode: str, optional
        :return: opened resource
        :rtype: file
        """

        puri = self._purge_uri(uri)
        puri_path = self._base_folder / Path(puri)
        puri_path.parent.mkdir(parents=True, exist_ok=True)
        return open(puri_path, mode)

    def flag(self, uri: str, flag: int):
        puri = self._purge_uri(uri)
        puri_path = self._base_folder / Path(puri)
        os.chflags(str(puri_path), flags=flag)

    def delete(self, uri: str):
        """ Deletes target resource

        :param uri: resource uri
        :type uri: str
        """

        puri = self._purge_uri(uri)
        puri_path: Path = self._base_folder / Path(puri)
        puri_path.unlink()

    @classmethod
    def create_from_configuration_file(cls, filename):
        cfg = SafeFilesystemDriverCFG(filename=filename)
        return SafeFilesystemDriver(cfg=cfg)
