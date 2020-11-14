
import numpy as np
from persefone.data.databases.filesystem.underfolder import UnderfolderLazySample
import dictquery as dq
from typing import List, Sequence, Union
from abc import ABC

from tqdm import tqdm


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
