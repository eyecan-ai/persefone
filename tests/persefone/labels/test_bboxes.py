from persefone.labels.bboxes import BoundingBox, BoundingBoxType
import pytest


class TestBoundingBoxes(object):

    @pytest.fixture
    def sample_bboxes(self):

        return [
            [True, 0, 0, 100, 100],
            [True, 100, 100, 200, 200],
            [True, -100, -100, -50, -50],
            [True, -100, -100, 300, 300],
            [False, -100, -100, 300],
            [False, -100, -100, 300, 2, 3, 4]
        ]

    def test_bboxes(self, sample_bboxes):

        ref_image_size = [1000, 1000]
        for sample in sample_bboxes:

            valid, data = sample[0], sample[1:]
            if valid:
                bbox = BoundingBox(data)

                for box_type in BoundingBoxType:
                    plain_data = bbox.plain_data(ref_image_size=ref_image_size, box_type=box_type)
                    bbox_r = BoundingBox.build_from_type(plain_data, box_type, image_size=ref_image_size)
                    assert bbox == bbox_r

            else:
                with pytest.raises(Exception):
                    print(data)
                    bbox = BoundingBox(data)
