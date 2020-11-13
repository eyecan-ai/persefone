
import dictquery as dq
from typing import List, Sequence, Union
from abc import ABC


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

        output = {}
        sample = self._dataset[idx]
        for key, v in sample.items():
            if key in self._keys_map:
                output[self._keys_map[key]] = v

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

    def __init__(self, queries: Sequence[str]):
        """ Query stage to filter samples based on specific
        queriable field (dict fields)

        :param queries: list of 'dictquery' queries
        :type queries: Sequence[str]
        """

        super().__init__()
        self._queries = queries

    def _update(self):

        self._indices = []

        for idx in range(len(self._dataset)):
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
