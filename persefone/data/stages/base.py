import random
from typing import List, Sequence, Union
from abc import ABC

import numpy as np
import dictquery as dq
from tqdm import tqdm

from persefone.data.databases.filesystem.underfolder import UnderfolderLazySample


class DStage(ABC):

    def __init__(self):
        """ Generic Dataset Stage """
        self._dataset = []

    def __call__(self, dataset: Sequence) -> 'DStage':
        """ Call stage pushing dataset inside

        :param dataset: input dataset
        :type dataset: Sequence
        :return: self stage
        :rtype: DStage
        """

        self._dataset = dataset
        self._update()
        return self

    def _update(self):
        """ update event called every dataset push.
        Each stage should implement its own 'update' event based
        on internal business logic.
        """
        pass

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx) -> dict:

        if idx >= len(self):
            raise IndexError

        return self._dataset[idx]


class StagesComposition(object):

    def __init__(self, stages: List[DStage]):
        """ Composition of several generic DStages

        :param stages: list of stages to be concat
        :type stages: List[DStage]
        """
        self._stages = stages

    def __call__(self, dataset: Sequence):
        for stage in self._stages:
            dataset = stage(dataset)
        return dataset


class StageKeyFiltering(DStage):

    def __init__(self, keys_map: Union[Sequence[str], List]):
        """ Stage for keys filtering

        :param keys_map: list of keys to be filtered or a dict representing also
        an implicit names remap
        :type keys_map: Union[Sequence[str], List]
        """
        super().__init__()

        self._keys_map = keys_map
        if isinstance(self._keys_map, list):
            self._keys_map = {k: k for k in self._keys_map}

    def __getitem__(self, idx) -> dict:

        if idx >= len(self):
            raise IndexError

        output = self._dataset[idx].copy()
        keys = list(output.keys())
        for key in keys:
            if key in self._keys_map:
                output[self._keys_map[key]] = output.pop(key)
            else:
                del output[key]

        return output


class StageSubsampling(DStage):

    def __init__(self, factor: int):
        """ Subsampling stage

        :param factor: subsampling factor
        :type factor: int
        """
        super().__init__()
        self._factor = factor
        self._indices = []

    def _update(self):
        self._indices = list(range(len(self._dataset)))
        self._indices = self._indices[::self._factor]

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx) -> dict:

        if idx >= len(self):
            raise IndexError

        return self._dataset[self._indices[idx]]


class StageQuery(DStage):

    def __init__(self, queries: Sequence[str], debug: bool = False):
        """ Query stage to filter samples based on specific
        queriable field (dict fields)

        :param queries: list of 'dictquery' queries
        :type queries: Sequence[str]
        """

        super().__init__()
        self._queries = queries
        self._debug = debug

    def _update(self):

        self._indices = []

        samples_iterator = range(len(self._dataset))
        if self._debug:
            samples_iterator = tqdm(samples_iterator)
            samples_iterator.set_description("Querying Dataset:")

        for idx in samples_iterator:
            sample = self._dataset[idx]
            matches = [dq.match(sample, q) for q in self._queries]
            if all(matches):
                self._indices.append(idx)

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx) -> dict:

        if idx >= len(self):
            raise IndexError

        return self._dataset[self._indices[idx]]


class StageCache(DStage):
    """Performs no operation on input dataset, but caches it to avoid recomputing previous expensive operations

    :param max_size: maximum cache size, set to a negative integer for unlimited cache, defaults to -1
    :type max_size: int, optional
    """

    def __init__(self, max_size: int = -1) -> None:
        super().__init__()
        self.max_size = max_size
        self._cache = {}

    def __getitem__(self, idx) -> dict:
        if idx in self._cache:
            return self._cache[idx]
        else:
            res = self._dataset[idx]
            if self.max_size < 0 or len(self._cache) < self.max_size:
                self._cache[idx] = res
            return res


class StageGroupBy(DStage):

    def __init__(self, field_name: str, stack_values: bool = False, debug: bool = False):

        super().__init__()
        self._field_name = field_name
        self._debug = debug
        self._stack_values = stack_values
        self._groups_ids = []

    def _update(self):

        self._indices = []

        samples_iterator = range(len(self._dataset))
        if self._debug:
            samples_iterator = tqdm(samples_iterator)
            samples_iterator.set_description("Grouping Dataset:")

        groups = {}

        fields_chunks = self._field_name.replace('`', '').split('.')
        for idx in samples_iterator:
            sample = self._dataset[idx]

            pointer = sample
            found = True
            for c in fields_chunks:
                try:
                    pointer = pointer[c]
                except KeyError:
                    found = False
                    break

            if found:
                if pointer not in groups:
                    groups[pointer] = []

                groups[pointer].append(idx)

        self._groups_ids = [(group_value, indices) for group_value, indices in groups.items()]

    def __len__(self):
        return len(self._groups_ids)

    def __getitem__(self, idx) -> dict:

        if idx >= len(self):
            raise IndexError

        group_value, group_indices = self._groups_ids[idx]

        output = None
        remaps = {}
        keys_to_remove = set()
        for idx in group_indices:
            sample = self._dataset[idx]

            if output is None:  # Copy From Sample to build a LazySample if needed
                output = sample.copy()

            for key in sample.keys():
                if key not in remaps:
                    remaps[key] = []
                remaps[key].append((key, idx))
                keys_to_remove.add(key)

        for key in keys_to_remove:
            del output[key]

        if not self._stack_values:
            for key, pairs in remaps.items():
                for k, i in pairs:
                    output[f'{k}_{i}'] = self._dataset[i].pop(k)
        else:
            for key, pairs in remaps.items():
                stack = []
                for k, i in pairs:
                    stack.append(self._dataset[i].pop(k))
                output[key] = stack
                if not isinstance(output, UnderfolderLazySample):
                    output[key] = np.stack(output[key])

        return output


class StageSlice(DStage):
    """Stage for sample selection based on indices

    :param start: start index, if None no constraint is applied, defaults to None
    :type start: int, optional
    :param stop: stop index (excluded), if None no constraint is applied, defaults to None
    :type stop: int, optional
    :param step: sample once every k elements, if None no subsampling is applied, defaults to None
    :type step: int, optional
    :param indices: select only specified indices (order applies), defaults to None
    :type indices: Sequence[int], optional

    Use start, stop and step to select a sample slice from the dataset:

    :example:
        >>> dataset = [{'a': i} for i in range(100)]
        >>> stage = StageSlice(0, 10, 2)
        >>> staged_dataset = stage(dataset)
        >>> list(staged_dataset)
        [{'a': 0}, {'a': 2}, {'a': 4}, {'a': 6}, {'a': 8}]

    You can also index the dataset with a custom list of indices:

    :example:
        >>> dataset = [{'a': i} for i in range(100)]
        >>> stage = StageSlice(indices=[1, 1, 3, 2, 2])
        >>> staged_dataset = stage(dataset)
        >>> list(staged_dataset)
        [{'a': 1}, {'a': 1}, {'a': 3}, {'a': 2}, {'a': 2}]
    """

    def __init__(self, start: int = None, stop: int = None, step: int = None, indices: Sequence[int] = None):
        super().__init__()
        self._start = start
        self._stop = stop
        self._step = step
        self._requested_indices = indices
        self._indices = []

    def _update(self):
        self._indices = list(range(len(self._dataset)))
        self._indices = self._indices[self._start:self._stop:self._step]
        if self._requested_indices is not None:
            self._indices = [x for x in self._requested_indices if x in self._indices]

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx) -> dict:

        if idx >= len(self):
            raise IndexError

        return self._dataset[self._indices[idx]]


class StageShuffle(DStage):
    """Stage for random shuffling dataset samples

    Use StageShuffle to randomly shuffle dataset samples:

    :example:
        >>> dataset = [{'a': i} for i in range(5)]
        >>> stage = StageShuffle()
        >>> staged_dataset = stage(dataset)
        >>> list(staged_dataset)
        [{'a': 1}, {'a': 0}, {'a': 4}, {'a': 3}, {'a': 2}]
    """

    def __init__(self):
        super().__init__()
        self._indices = []

    def _update(self):
        self._indices = list(range(len(self._dataset)))
        random.shuffle(self._indices)

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx) -> dict:

        if idx >= len(self):
            raise IndexError

        return self._dataset[self._indices[idx]]
