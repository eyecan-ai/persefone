from pathlib import Path
from collections import defaultdict
import imghdr
import numpy as np


def tree_from_underscore_notation_files(folder, skip_hidden_files=True):
    """Walk through files in folder generating a tree based on Underscore notation.
    Leafs of abovementioned tree are string representing filenames, inner nodes represent
    keys hierarchy.

    :param folder: target folder
    :type folder: str
    :param skip_hidden_files: TRUE to skip files starting with '.', defaults to True
    :type skip_hidden_files: bool, optional
    :return: dictionary representing multilevel tree
    :rtype: dict
    """

    # Declare TREE structure
    def tree():
        return defaultdict(tree)

    keys_tree = tree()
    folder = Path(folder)
    files = list(sorted(folder.glob('*')))
    for f in files:
        name = f.stem

        if skip_hidden_files:
            if name.startswith('.'):
                continue

        chunks = name.split('_', maxsplit=1)
        if len(chunks) == 1:
            chunks.append('none')
        p = keys_tree
        for index, chunk in enumerate(chunks):
            if index < len(chunks) - 1:
                p = p[chunk]
            else:
                p[chunk] = str(f)

    return dict(keys_tree)


def get_file_extension(filename, with_dot=False):
    ext = Path(filename).suffix.lower()
    if not with_dot and len(ext) > 0:
        ext = ext[1:]
    return ext


def is_file_image(filename):
    return imghdr.what(filename) is not None


def is_file_numpy_array(filename):
    ext = get_file_extension(filename)
    if ext in ['txt', 'data']:
        try:
            np.loadtxt(filename)
            return True
        except Exception:
            return False
    if ext in ['npy', 'npz']:
        try:
            np.load(filename)
            return True
        except Exception:
            return False
    return False
