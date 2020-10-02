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
                'data': [3, 0.34, 0.15, 0.23, 0.14]
            },
            {
                'fmt': 'c' + FieldsOptions.get_format(FieldsOptions.FORMAT_ALBUMENTATIONS),
                'data': [3, 0.34, 0.45, 0.44, 0.64]
            }
        ]

    @pytest.fixture
    def sample_sizes(self):
        return [
            [1000, 1000],
            [100, 100],
            [345, 212],
            [111, 500],
        ]

    def test_bboxes(self, sample_data, sample_formats, sample_sizes):

        for ref_image_size in sample_sizes:

            for sample in sample_data:

                label = BoundingBoxLabel(data=sample['data'], fmt=sample['fmt'], image_size=ref_image_size)
                print("#"*20, sample['fmt'])
                print(sample['data'])
                print(label.plain_data(fmt=sample['fmt']))
                assert np.all(np.isclose(sample['data'], label.plain_data(fmt=sample['fmt']), rtol=0.01))
                original_athoms = label.athoms()

                for fmt in sample_formats:
                    print(sample['fmt'], fmt, sample['data'])
                    relabel = BoundingBoxLabel(data=label.plain_data(fmt), fmt=fmt, image_size=ref_image_size)
                    reathoms = relabel.athoms()

                    for f, v in original_athoms.items():
                        assert np.isclose(original_athoms[f], reathoms[f], rtol=0.01), f
