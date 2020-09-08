from persefone.labels.bboxes import BoundingBox, BoundingBoxType, BoundingBoxWithLabelAndScore
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

    @pytest.fixture
    def sample_bboxes_with_score_and_label(self):

        return [
            [True, 0, 0, 100, 100, 0.5, 4],
            [True, 100, 100, 200, 200, 0.5, 4.0],
            [True, -100, -100, -50, -50, 0.5, 4.0],
            [True, -100, -100, 300, 300, 0.5, 4.0],
            [False, -100, -100, 300, 0.5, 4.0],
            [False, -100, -100, 300, 0.5, 4.0, 2, 3, 4]
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

                    bbox_d = BoundingBox.from_dict(bbox_r.as_dict(ref_image_size=ref_image_size))
                    assert bbox_d == bbox

            else:
                with pytest.raises(Exception):
                    bbox = BoundingBox(data)

    def test_bboxes_with_score_and_labels(self, sample_bboxes_with_score_and_label):

        ref_image_size = [1000, 1000]
        for sample in sample_bboxes_with_score_and_label:

            valid, data = sample[0], sample[1:]
            if valid:
                print("DATA", data)
                bbox = BoundingBoxWithLabelAndScore(data)

                for box_type in BoundingBoxType:
                    plain_data = bbox.plain_data(ref_image_size=ref_image_size, box_type=box_type)
                    bbox_r = BoundingBoxWithLabelAndScore.build_from_type(plain_data, box_type, image_size=ref_image_size)
                    assert bbox == bbox_r

                    bbox_d = BoundingBoxWithLabelAndScore.from_dict(bbox_r.as_dict(ref_image_size=ref_image_size))
                    assert bbox_d == bbox

            else:
                with pytest.raises(Exception):
                    print(data)
                    bbox = BoundingBoxWithLabelAndScore(data)
