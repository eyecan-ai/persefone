
from pathlib import Path
from typing import Dict, Union
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

        # builds tree from folder with underscore notation
        self._tree = tree_from_underscore_notation_files(self._folder / self.DATA_SUBFOLDER)

        self._ids = list(sorted(self._tree.keys()))

        self._dataset_files = [x for x in Path(self._folder).glob('*') if x.is_file()]
        self._dataset_metadata = {}
        for f in self._dataset_files:
            self._dataset_metadata[f.stem] = self._load_data(f)

    @property
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

    def _load_data(self, filename: str) -> Union[None, np.ndarray, dict]:
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

    def __getitem__(self, idx):

        if idx >= len(self):
            raise IndexError

        data = self._tree[self._ids[idx]]
        output = {}
        for tag, filename in data.items():
            remap = self._get_tag_remap(tag)
            if remap is not None:
                output[remap] = self._load_data(filename)

        return output
