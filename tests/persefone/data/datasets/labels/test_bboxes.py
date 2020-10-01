from persefone.data.datasets.labels.bboxes import FieldsOptions, BoundingBoxLabel
import pytest

import numpy as np


class TestGeometricLabels(object):

    @pytest.fixture
    def sample_formats(self):
        return [
            'c' + FieldsOptions.get_format(FieldsOptions.FORMAT_PASCAL_VOC),
            'c' + FieldsOptions.get_format(FieldsOptions.FORMAT_COCO),
            'c' + FieldsOptions.get_format(FieldsOptions.FORMAT_ALBUMENTATIONS),
            'c' + FieldsOptions.get_format(FieldsOptions.FORMAT_YOLO),
        ]

    @pytest.fixture
    def sample_data(self):

        return [
            {
                'fmt': 'c' + FieldsOptions.get_format(FieldsOptions.FORMAT_PASCAL_VOC),
                'data': [3, 24., 32., 114., 251.]
            },
            {
                'fmt': 'c' + FieldsOptions.get_format(FieldsOptions.FORMAT_COCO),
                'data': [3, 24., 32, 50., 50.]
            },
            {
                'fmt': 'c' + FieldsOptions.get_format(FieldsOptions.FORMAT_YOLO),
                'data': [3, 0.34, 0.45, 0.23, 0.24]
            },
            {
                'fmt': 'c' + FieldsOptions.get_format(FieldsOptions.FORMAT_ALBUMENTATIONS),
                'data': [3, 0.34, 0.45, 0.44, 0.64]
            }
        ]

    def test_bboxes(self, sample_data, sample_formats):

        ref_image_size = [1000, 1000]

        for sample in sample_data:

            label = BoundingBoxLabel(data=sample['data'], fmt=sample['fmt'], image_size=ref_image_size)
            original_athoms = label.athoms()

            for fmt in sample_formats:
                print(sample['fmt'], fmt, sample['data'])
                relabel = BoundingBoxLabel(data=label.plain_data(fmt), fmt=fmt, image_size=ref_image_size)
                reathoms = relabel.athoms()

                for f, v in original_athoms.items():
                    assert np.isclose(original_athoms[f], reathoms[f]), f
