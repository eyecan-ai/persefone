from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, average_precision_score
import numpy as np
import logging
from deprecated import deprecated

_logger = logging.getLogger(__file__)


class BinaryClassificationMetrics(object):

    def __init__(self, gt, scores):
        self.gt = np.array(gt)
        self.scores = np.array(scores)
        assert len(self.gt) == len(self.scores), "Len of GT / SCORES must be the same!"

    def precision_recall_curve(self, with_thresholds=False) -> dict:
        """Computes Precision/Recall curve as dict of lists

        :param with_thresholds: TRUE to compute also 'thresholds' values, defaults to False
        :type with_thresholds: bool, optional
        :return: dict with 'precision' and 'recall' lists (or even 'thresholds' list)
        :rtype: dict
        """
        precision, recall, thresholds = precision_recall_curve(y_true=self.gt, probas_pred=self.scores)
        output = {
            'precision': precision,
            'recall': recall
        }
        if with_thresholds:
            output['thresholds'] = thresholds
        return output

    def ap(self) -> float:
        """Computes Average Precision

        :return: average precision of gt/scores
        :rtype: float
        """
        try:
            return average_precision_score(y_true=self.gt, y_score=self.scores)
        except Exception as e:
            _logger.error(e)
            return 0.0

    def roc_curve(self, with_thresholds=False) -> dict:
        """Computes ROC curve as dict of lists

        :param with_thresholds: TRUE to compute also 'thresholds' values, defaults to False
        :type with_thresholds: bool, optional
        :return: dict with 'tpr' and 'fpr' lists (or even 'thresholds' list)
        :rtype: dict
        """

        fpr, tpr, thresholds = roc_curve(y_true=self.gt, y_score=self.scores)

        output = {
            'fpr': fpr,
            'tpr': tpr,
        }
        if with_thresholds:
            output['thresholds'] = thresholds
        return output

    def auc(self) -> float:
        """Computes single AUC score

        :return: auc score
        :rtype: float
        """
        try:
            return roc_auc_score(y_true=self.gt, y_score=self.scores)
        except Exception as e:
            _logger.error(e)
            return 0.0

    def compute_metrics(self, th=0.5) -> dict:
        """Computes a bunch of metrics based on a target threshold

        :param th: target threshold, defaults to 0.5
        :type th: float, optional
        :return: dict of metrics
        :rtype: dict
        """

        preds = np.zeros_like(self.scores)
        preds[self.scores >= th] = 1
        preds[self.scores < th] = 0

        cm = confusion_matrix(self.gt, preds).ravel()
        if len(cm) == 1:
            cm = np.array([0, 0, 0, np.asscalar(cm)])

        tn, fp, fn, tp = cm
        tn_perc, fp_perc, fn_perc, tp_perc = cm / self.gt.shape[0]

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = TPR = sensitivity = tp / (tp + fn)
        specificity = TNR = tn / (tn + fp)
        precision = PPV = tp / (tp + fp)
        missrate = FNR = fn / (fn + tp)
        FPR = fp / (fp + tn)
        balanced_accuracy = (TPR + TNR) / 2.
        f1_score = 2 * (PPV * TPR) / (PPV + TPR)

        metrics = {
            "tn": tn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn_perc": tn_perc,
            "tp_perc": tp_perc,
            "fp_perc": fp_perc,
            "fn_perc": fn_perc,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "recall": recall,
            "sensitivity": sensitivity,
            "TPR": TPR,
            "specificity": specificity,
            "TNR": TNR,
            "precision": precision,
            "PPV": PPV,
            "missrate": missrate,
            "FNR": FNR,
            "fallout": FPR,
            "FPR": FPR,
            "f1_score": f1_score
        }

        return metrics


@deprecated(reason="use BinaryClassificationMetrics")
class BinaryClassifcationMetrics(BinaryClassificationMetrics):

    def __init__(self, gt, scores):
        super(BinaryClassifcationMetrics, self).__init__(gt, scores)
