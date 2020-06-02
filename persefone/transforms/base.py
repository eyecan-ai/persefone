from abc import ABC


class AbstractFactory(ABC):

    @classmethod
    def _targets_map(cls):
        return {
            'pixels': ['image', 'mask', 'masks', 'bboxes', 'keypoints'],
            'spatial_full': ['image', 'mask', 'masks', 'bboxes', 'keypoints'],
            'spatial_half': ['image', 'mask', 'masks'],
        }

    @classmethod
    def build_transform(cls, name, **params):
        return None
