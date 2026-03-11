import numpy as np
from spectra.metrics import accuracy, classification_report, confusion_matrix, per_snr_accuracy


class TestConfusionMatrix:
    def test_perfect_predictions(self):
        y = [0, 1, 2, 0, 1, 2]
        cm = confusion_matrix(y, y, num_classes=3)
        assert cm.shape == (3, 3)
        assert np.array_equal(cm, np.diag([2, 2, 2]))

    def test_all_wrong(self):
        y_true = [0, 0, 0]
        y_pred = [1, 1, 1]
        cm = confusion_matrix(y_true, y_pred, num_classes=2)
        assert cm[0, 1] == 3
        assert cm[0, 0] == 0

    def test_known_answer(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 1, 2, 2, 0]
        cm = confusion_matrix(y_true, y_pred, num_classes=3)
        assert cm[0, 0] == 1
        assert cm[0, 1] == 1
        assert cm[1, 1] == 1
        assert cm[1, 2] == 1
        assert cm[2, 2] == 1
        assert cm[2, 0] == 1


class TestAccuracy:
    def test_perfect(self):
        assert accuracy([0, 1, 2], [0, 1, 2]) == 1.0

    def test_half(self):
        assert accuracy([0, 0, 1, 1], [0, 1, 0, 1]) == 0.5

    def test_empty(self):
        assert accuracy([], []) == 0.0


class TestClassificationReport:
    def test_perfect_report(self):
        y = [0, 0, 1, 1]
        report = classification_report(y, y, class_names=["A", "B"])
        assert report["A"]["precision"] == 1.0
        assert report["A"]["recall"] == 1.0
        assert report["A"]["f1"] == 1.0
        assert report["A"]["support"] == 2

    def test_known_answer(self):
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [0, 0, 1, 1, 1, 0]
        report = classification_report(y_true, y_pred)
        # Class 0: TP=2, FP=1, FN=1 -> precision=2/3, recall=2/3
        assert abs(report["0"]["precision"] - 2 / 3) < 1e-6
        assert abs(report["0"]["recall"] - 2 / 3) < 1e-6

    def test_no_class_names(self):
        report = classification_report([0, 1], [0, 1])
        assert "0" in report
        assert "1" in report


class TestPerSNRAccuracy:
    def test_grouped_accuracy(self):
        y_true = [0, 0, 1, 1, 0, 0]
        y_pred = [0, 0, 1, 0, 0, 1]
        snrs = [0, 0, 0, 10, 10, 10]
        result = per_snr_accuracy(y_true, y_pred, snrs)
        assert result[0.0] == 1.0  # all correct at SNR=0
        assert abs(result[10.0] - 1 / 3) < 1e-6  # 1 of 3 at SNR=10

    def test_single_snr(self):
        result = per_snr_accuracy([0, 1], [0, 0], [5.0, 5.0])
        assert result[5.0] == 0.5
