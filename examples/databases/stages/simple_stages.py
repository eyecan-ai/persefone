import time
from persefone.data.databases.filesystem.underfolder import UnderfolderDatabase
from persefone.data.stages.base import (
    StageKeyFiltering,
    StagesComposition,
    StageSubsampling,
    StageQuery
)
from persefone.data.stages.transforms import StageTransforms
import cv2

from contextlib import contextmanager
from time import time


@contextmanager
def timing(description: str) -> None:
    start = time()
    yield
    ellapsed_time = time() - start
    print(f"{description}: {ellapsed_time}")


folder = '../../../tests/sample_data/datasets/underfolder'
database = UnderfolderDatabase(folder)

stages = StagesComposition([
    StageKeyFiltering([
        'metadata',
        'image',
        'image_mask',
        'image_maskinv'
    ]),
    StageSubsampling(factor=2),
    StageQuery(queries=[
        '`metadata.sample_id` >= 2',
        '`metadata.sample_id` < 12'
    ]),
    StageTransforms(augmentations='transforms.yml'),
    StageTransforms(augmentations='augmentations.yml'),
    StageKeyFiltering({
        'metadata': 'info',
        'image': 'x',
        'image_mask': 'gt',
        'image_maskinv': 'gt_inv'
    }),
    StageQuery(queries=[
        '`info.sample_id` < 111',
    ]),
])

with timing("Staging database"):
    database = stages(database)


while len(database) > 0:
    for sample_id in range(len(database)):
        sample = database[sample_id]
        print(sample['info'])
        cv2.imshow("x", sample['x'])
        cv2.imshow("gt", sample['gt'])
        cv2.imshow("gt_inv", sample['gt_inv'])
        if ord('q') == cv2.waitKey(0):
            import sys
            sys.exit(0)
