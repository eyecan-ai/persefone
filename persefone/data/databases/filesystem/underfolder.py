
from json.encoder import JSONEncoder
from pathlib import Path
from persefone.utils.bytes import DataCoding
from typing import Any, Dict, Union
from persefone.utils.filesystem import tree_from_underscore_notation_files, is_file_image, get_file_extension, is_file_numpy_array, is_metadata_file
import imageio
import numpy as np
import yaml
import json


class UnderfolderDatabase(object):
    DATA_SUBFOLDER = 'data'

    def __init__(self, folder: str, data_tags: Union[None, Dict] = None):
        """ Creates a database based on an UNderscore Notation Folder

        :param folder: input folder
        :type folder: str
        :param data_tags: dict of allowed tags with remapped name, leave None for all tags allowed, defaults to None
        :type data_tags: Union[None, Dict], optional
        """

        self._folder = Path(folder)
        self._data_tags = data_tags

        self._data_folder = self._folder / self.DATA_SUBFOLDER

        if self._data_folder.exists():
            # builds tree from subfolder with underscore notation
            self._tree = tree_from_underscore_notation_files(self._folder / self.DATA_SUBFOLDER)
            self._ids = list(sorted(self._tree.keys()))

            self._dataset_files = [x for x in Path(self._folder).glob('*') if x.is_file()]
            self._dataset_metadata = {}
            for f in self._dataset_files:
                self._dataset_metadata[f.stem] = self.load_data(f)
        else:
            # builds tree from current folder with underscore notation
            self._data_folder = self._folder
            self._tree = tree_from_underscore_notation_files(self._folder)
            self._ids = list(sorted(self._tree.keys()))
            self._dataset_files = []
            self._dataset_metadata = {}

        self._filenames = [None] * len(self)
        for idx in range(len(self)):
            self._filenames[idx] = self._get_filenames(idx)

    @property
    def base_folder(self):
        return self._folder

    @property
    def data_folder(self):
        return self._data_folder

    @property
    def has_data_in_subfolder(self):
        return self.base_folder != self.data_folder

    @ property
    def metadata(self):
        return self._dataset_metadata

    def _get_tag_remap(self, tag: str) -> Union[str, None]:
        """ Returns the remapped tag name. If data_tags list is None
        returns the same name. May return None if tag is not present

        :param tag: input tag
        :type tag: str
        :return: remapped tag name or None
        :rtype:  Union[str, None]
        """

        if self._data_tags is None:
            return tag
        return self._data_tags.get(tag, None)

    def load_data(self, filename: str) -> Union[None, np.ndarray, dict]:
        """ Load data from file based on its extension

        :param filename: target filename
        :type filename: str
        :return: Loaded data as array or dict. May return NONE
        :rtype: Union[None, np.ndarray, dict]
        """

        extension = get_file_extension(filename)
        data = None

        if is_file_image(filename):
            data = imageio.imread(filename)

        if is_file_numpy_array(filename):
            if extension in ['txt']:
                data = np.loadtxt(filename)
            elif extension in ['npy', 'npz']:
                data = np.load(filename)
            if data is not None:
                data = np.atleast_2d(data)

        if is_metadata_file(filename):
            if extension in ['yml']:
                data = yaml.safe_load(open(filename, 'r'))
            elif extension in ['json']:
                data = json.load(open(filename))

        return data

    def __len__(self):
        return len(self._ids)

    def _get_filenames(self, idx):

        if idx >= len(self):
            raise IndexError

        data = self._tree[self._ids[idx]]
        output = {}
        for tag, filename in data.items():
            remap = self._get_tag_remap(tag)
            output[remap] = str(filename)

        output['_id'] = self._ids[idx]
        return output

    @property
    def skeleton(self):
        return self._filenames

    def __getitem__(self, idx):

        if idx >= len(self):
            raise IndexError

        data = self._tree[self._ids[idx]]
        output = {}
        for tag, filename in data.items():
            remap = self._get_tag_remap(tag)
            if remap is not None:
                output[remap] = self.load_data(filename)

        return output


class UnderfolderDatabaseGenerator(object):

    class NumpyArrayEncoder(JSONEncoder):

        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)

    def __init__(self, folder: str, zfill: int = 5, create_if_none: bool = True):
        self._folder = Path(folder)
        self._data_folder = self._folder / UnderfolderDatabase.DATA_SUBFOLDER
        if create_if_none:
            self._data_folder.mkdir(exist_ok=True, parents=True)
        self._zfill = zfill

    def _generate_number(self, idx: int):
        return str(idx).zfill(self._zfill)

    def _generate_tag_name(self, idx: int, tag: str):
        return f'{self._generate_number(idx)}_{tag}'

    def _purge_metadata(self, metadata: dict):
        return json.loads(json.dumps(metadata, cls=self.NumpyArrayEncoder))

    def store_sample(self, idx: int, tag: str, data: Any, extension: str) -> str:
        """ Stores sample computing name based on IDX/TAG and extension

        :param idx: sample id
        :type idx: int
        :param tag: item tag
        :type tag: str
        :param data: data to store
        :type data: Any
        :param extension: desired extension
        :type extension: str
        :raises NotImplementedError: not supported extension
        """

        extension = extension.replace('.', '')
        if idx >= 0:
            name = self._generate_tag_name(idx, tag)
            filename = self._data_folder / f'{name}.{extension}'
        else:
            filename = self._folder / f'{tag}.{extension}'

        if extension in DataCoding.IMAGE_CODECS:
            imageio.imwrite(filename, data)
        elif extension in DataCoding.NUMPY_CODECS:
            np.save(filename, data)
        elif extension in DataCoding.TEXT_CODECS:
            np.savetxt(filename, data)
        elif extension in DataCoding.METADATA_CODECS:
            if 'json' in extension:
                json.dump(self._purge_metadata(data), open(filename, 'w'))
            elif 'yml' in extension or 'yaml':
                yaml.safe_dump(self._purge_metadata(data), open(filename, 'w'))
            else:
                raise NotImplementedError(f"EXtension [{extension}] not supported as metadata yet!")
        else:
            raise NotImplementedError(f"EXtension [{extension}] not supported yet!")

        return str(filename)
