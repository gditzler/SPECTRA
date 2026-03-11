"""Tests for CyclostationaryAMC classifier."""

import numpy as np
import pytest
from spectra.classifiers.amc import CyclostationaryAMC
from spectra.datasets.cyclo import CyclostationaryDataset
from spectra.transforms import Cumulants
from spectra.waveforms import BPSK, QAM16, QPSK

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_iq(waveform, n_iq=4096, sample_rate=1e6, seed=0):
    sps = getattr(waveform, "samples_per_symbol", 8)
    num_symbols = n_iq // sps + 1
    iq = waveform.generate(num_symbols=num_symbols, sample_rate=sample_rate, seed=seed)
    return iq[:n_iq]


def _make_training_data(waveforms, n_per_class=20, n_iq=4096, sample_rate=1e6):
    """Generate labeled IQ data for multiple waveform classes."""
    iq_list = []
    labels = []
    for label_idx, wf in enumerate(waveforms):
        for seed in range(n_per_class):
            iq_list.append(_generate_iq(wf, n_iq=n_iq, sample_rate=sample_rate, seed=seed))
            labels.append(label_idx)
    return iq_list, np.array(labels)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


@pytest.mark.csp
class TestFeatureExtraction:
    def test_cumulant_features_shape(self):
        amc = CyclostationaryAMC(feature_set="cumulants")
        iq = _generate_iq(BPSK())
        features = amc.extract_features(iq)
        assert features.shape == (5,)
        assert features.dtype == np.float32

    def test_cyclic_peaks_features_shape(self):
        amc = CyclostationaryAMC(feature_set="cyclic_peaks")
        iq = _generate_iq(BPSK())
        features = amc.extract_features(iq)
        assert features.shape == (15,)  # 5 peaks * 3 values each
        assert features.dtype == np.float32

    def test_combined_features_shape(self):
        amc = CyclostationaryAMC(feature_set="combined")
        iq = _generate_iq(BPSK())
        features = amc.extract_features(iq)
        assert features.shape == (20,)  # 5 cumulants + 15 cyclic peaks
        assert features.dtype == np.float32

    def test_features_no_nan_inf(self):
        amc = CyclostationaryAMC(feature_set="combined")
        for wf in [BPSK(), QPSK(), QAM16()]:
            iq = _generate_iq(wf)
            features = amc.extract_features(iq)
            assert not np.any(np.isnan(features)), f"NaN in {wf.label}"
            assert not np.any(np.isinf(features)), f"Inf in {wf.label}"

    def test_features_nonnegative(self):
        """Cumulant magnitudes and normalised peak values are >= 0."""
        amc = CyclostationaryAMC(feature_set="combined")
        iq = _generate_iq(BPSK())
        features = amc.extract_features(iq)
        assert np.all(features >= 0.0)

    def test_different_signals_produce_different_features(self):
        amc = CyclostationaryAMC(feature_set="cumulants")
        f_bpsk = amc.extract_features(_generate_iq(BPSK(), seed=0))
        f_qpsk = amc.extract_features(_generate_iq(QPSK(), seed=0))
        assert not np.allclose(f_bpsk, f_qpsk)


# ---------------------------------------------------------------------------
# Fit / predict cycle
# ---------------------------------------------------------------------------


@pytest.mark.csp
class TestFitPredict:
    def test_fit_predict_decision_tree(self):
        amc = CyclostationaryAMC(feature_set="cumulants", classifier="decision_tree")
        waveforms = [BPSK(), QPSK()]
        iq_list, labels = _make_training_data(waveforms, n_per_class=15)
        X = np.stack([amc.extract_features(iq) for iq in iq_list])
        amc.fit(X, labels)
        preds = amc.predict(X)
        assert preds.shape == labels.shape
        # Should achieve >50% on training data (not random)
        accuracy = (preds == labels).mean()
        assert accuracy > 0.5

    def test_fit_predict_random_forest(self):
        amc = CyclostationaryAMC(feature_set="cumulants", classifier="random_forest")
        waveforms = [BPSK(), QPSK()]
        iq_list, labels = _make_training_data(waveforms, n_per_class=15)
        X = np.stack([amc.extract_features(iq) for iq in iq_list])
        amc.fit(X, labels)
        preds = amc.predict(X)
        accuracy = (preds == labels).mean()
        assert accuracy > 0.5

    def test_predict_before_fit_raises(self):
        amc = CyclostationaryAMC()
        with pytest.raises(RuntimeError, match="not been fitted"):
            amc.predict(np.zeros((5, 5)))

    def test_fit_returns_self(self):
        amc = CyclostationaryAMC(feature_set="cumulants")
        iq_list, labels = _make_training_data([BPSK()], n_per_class=5)
        X = np.stack([amc.extract_features(iq) for iq in iq_list])
        result = amc.fit(X, labels)
        assert result is amc


# ---------------------------------------------------------------------------
# Dataset convenience method
# ---------------------------------------------------------------------------


@pytest.mark.csp
class TestFitFromDataset:
    def test_fit_from_dataset(self):
        ds = CyclostationaryDataset(
            waveform_pool=[BPSK(), QPSK()],
            num_samples=20,
            num_iq_samples=4096,
            sample_rate=1e6,
            representations={"cum": Cumulants(max_order=4)},
            seed=42,
        )
        amc = CyclostationaryAMC(feature_set="cumulants")
        amc.fit_from_dataset(ds)
        # Should be fitted now
        iq = _generate_iq(BPSK(), seed=99)
        features = amc.extract_features(iq).reshape(1, -1)
        pred = amc.predict(features)
        assert pred.shape == (1,)

    def test_fit_from_dataset_max_samples(self):
        ds = CyclostationaryDataset(
            waveform_pool=[BPSK(), QPSK()],
            num_samples=50,
            num_iq_samples=4096,
            sample_rate=1e6,
            representations={"cum": Cumulants(max_order=4)},
            seed=0,
        )
        amc = CyclostationaryAMC(feature_set="cumulants")
        amc.fit_from_dataset(ds, max_samples=10)
        # Should still work after fitting with fewer samples
        iq = _generate_iq(BPSK(), seed=99)
        features = amc.extract_features(iq).reshape(1, -1)
        pred = amc.predict(features)
        assert pred.shape == (1,)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.csp
class TestAMCValidation:
    def test_invalid_feature_set_raises(self):
        with pytest.raises(ValueError, match="Unknown feature_set"):
            CyclostationaryAMC(feature_set="bogus")

    def test_invalid_classifier_raises(self):
        with pytest.raises(ValueError, match="Unknown classifier"):
            CyclostationaryAMC(classifier="svm")


# ---------------------------------------------------------------------------
# Classification accuracy
# ---------------------------------------------------------------------------


@pytest.mark.csp
class TestClassificationAccuracy:
    def test_bpsk_vs_qpsk_accuracy(self):
        """BPSK and QPSK should be distinguishable by cumulants."""
        amc = CyclostationaryAMC(feature_set="cumulants", classifier="random_forest")
        waveforms = [BPSK(), QPSK()]
        train_iq, train_labels = _make_training_data(waveforms, n_per_class=30)
        X_train = np.stack([amc.extract_features(iq) for iq in train_iq])
        amc.fit(X_train, train_labels)

        # Test on held-out data (different seeds via offset)
        test_iq, test_labels = _make_training_data(waveforms, n_per_class=10, n_iq=4096)
        # Use seeds offset by 100 to avoid training overlap
        test_iq2 = []
        test_labels2 = []
        for label_idx, wf in enumerate(waveforms):
            for seed in range(100, 110):
                test_iq2.append(_generate_iq(wf, seed=seed))
                test_labels2.append(label_idx)
        test_labels2 = np.array(test_labels2)
        X_test = np.stack([amc.extract_features(iq) for iq in test_iq2])
        preds = amc.predict(X_test)
        accuracy = (preds == test_labels2).mean()
        assert accuracy >= 0.80, f"BPSK vs QPSK accuracy {accuracy:.2f} < 0.80"
