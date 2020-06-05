import pytest
from persefone.metrics.commons import BinaryClassifcationMetrics
import numpy as np

GT_SCORES_SAMPLES = [
    {
        'gt': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        'scores': [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5],
        'ap': 0.5,
        'auc': 0.5,
        'valid': True
    },
    {
        'gt': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'scores': [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5],
        'ap': 0.5,
        'auc': 0.5,
        'valid': True
    },
    {
        'gt': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'scores': [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5],
        'ap': 0.5,
        'auc': 0.5,
        'valid': True
    },
    {
        'gt': [0, 0, 0, 0, 1],
        'scores': [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5],
        'ap': 0.5,
        'auc': 0.5,
        'valid': False
    }
]


class TestBinaryClassifcationMetrics(object):

    @pytest.mark.filterwarnings("ignore")  # TODO: Metrics consistency strange behaviour -> lot of warnings
    @pytest.mark.parametrize("sample", GT_SCORES_SAMPLES)
    def test_consistency(self, sample):

        if sample['valid']:
            ths = np.arange(0.0, 1.01, 0.1)
            for th in ths:
                pm = BinaryClassifcationMetrics(
                    gt=sample['gt'],
                    scores=sample['scores']
                )

                pm.compute_metrics(th=th)

            print("AP", pm.ap())
            print("AUC", pm.auc())

            assert 'precision' in pm.precision_recall_curve(with_thresholds=False), "Precision list is missing"
            assert 'recall' in pm.precision_recall_curve(with_thresholds=False), "Recall list is missing"
            assert 'thresholds' not in pm.precision_recall_curve(with_thresholds=False), "Threshold list should be missing"
            assert 'thresholds' in pm.precision_recall_curve(with_thresholds=True), "Threshold list is missing"

            assert 'fpr' in pm.roc_curve(with_thresholds=False), "FPR list is missing"
            assert 'tpr' in pm.roc_curve(with_thresholds=False), "TPR list is missing"
            assert 'thresholds' not in pm.roc_curve(with_thresholds=False), "Threshold list should be missing"
            assert 'thresholds' in pm.roc_curve(with_thresholds=True), "Threshold list is missing"

            print("PR Curve", pm.precision_recall_curve())
        else:
            with pytest.raises(AssertionError):
                pm = BinaryClassifcationMetrics(
                    gt=sample['gt'],
                    scores=sample['scores']
                )
