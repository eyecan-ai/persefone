from persefone.utils.configurations import XConfiguration
from schema import Schema, Optional
import random
import numpy as np


class MongoSnapshotCFG(XConfiguration):

    def __init__(self, filename=None):
        XConfiguration.__init__(self, filename=filename)
        self.set_schema(Schema({
            # NAME
            Optional('_reusable'): dict,
            'datasets': [
                {str: {
                    'dataset': {'name': str},
                    'reader': {'data_mapping': {str: str},
                               Optional('queries'): [str],
                               Optional('orders'): [str]
                               }
                }
                }
            ],
            'pipeline': str
        }))


class SnapshotOperations(object):

    @classmethod
    def shuffle(cls, d, seed=-1):
        if seed > 0:
            random.seed(seed)
        random.shuffle(d)
        return d

    @classmethod
    def split(cls, d, percentage):
        percentage = np.clip(percentage, 0.0, 1.0)
        plen = int(len(d) * percentage)
        d0 = d[:plen]
        d1 = d[plen:]
        assert len(d0) + len(d1) == len(d), f"splits size not sum up to total size! {len(d0)} + {len(d1)} != {len(d)}"
        return d0, d1
