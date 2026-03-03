"""Traditional cyclostationary AMC classifier.

Uses higher-order cumulant features and/or cyclic spectral peaks
with scikit-learn tree-based classifiers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from spectra.transforms import Cumulants, SCD

if TYPE_CHECKING:
    from spectra.datasets.cyclo import CyclostationaryDataset


class CyclostationaryAMC:
    """Automatic Modulation Classifier using CSP features.

    Extracts hand-crafted features from IQ signals and classifies
    modulation type with a scikit-learn tree-based model.

    Args:
        feature_set: ``"cumulants"`` (5-dim |C20|..|C42|),
            ``"cyclic_peaks"`` (dominant SCD peak locations + magnitudes),
            or ``"combined"`` (both concatenated).
        classifier: ``"decision_tree"`` or ``"random_forest"``.
        scd_nfft: FFT size for SCD used in ``"cyclic_peaks"`` features.
        scd_n_alpha: Number of cyclic-frequency bins for SCD.

    Example::

        amc = CyclostationaryAMC(feature_set="cumulants")
        X = np.stack([amc.extract_features(iq) for iq in iq_signals])
        amc.fit(X, labels)
        predictions = amc.predict(X_test)
    """

    def __init__(
        self,
        feature_set: str = "cumulants",
        classifier: str = "decision_tree",
        scd_nfft: int = 64,
        scd_n_alpha: int = 64,
    ):
        if feature_set not in ("cumulants", "cyclic_peaks", "combined"):
            raise ValueError(
                f"Unknown feature_set: {feature_set!r}. "
                "Supported: 'cumulants', 'cyclic_peaks', 'combined'."
            )
        if classifier not in ("decision_tree", "random_forest"):
            raise ValueError(
                f"Unknown classifier: {classifier!r}. "
                "Supported: 'decision_tree', 'random_forest'."
            )
        self.feature_set = feature_set
        self.classifier_type = classifier
        self.scd_nfft = scd_nfft
        self.scd_n_alpha = scd_n_alpha

        self._cum = Cumulants(max_order=4)
        self._scd = SCD(nfft=scd_nfft, n_alpha=scd_n_alpha, hop=scd_nfft // 4)
        self._model = None

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _cumulant_features(self, iq: np.ndarray) -> np.ndarray:
        """5-dim cumulant magnitude vector: |C20|, |C21|, |C40|, |C41|, |C42|."""
        return self._cum(iq).numpy()

    def _cyclic_peak_features(self, iq: np.ndarray, n_peaks: int = 5) -> np.ndarray:
        """Top-*n_peaks* SCD magnitudes and their (f, alpha) indices.

        Returns a flat vector of length ``3 * n_peaks`` containing
        ``[mag_0, f_idx_0, alpha_idx_0, mag_1, ...]`` normalised to
        ``[0, 1]``.
        """
        scd_tensor = self._scd(iq)  # [1, Nf, Na]
        scd_mag = scd_tensor.squeeze(0).numpy()  # [Nf, Na]
        flat = scd_mag.ravel()
        # Take top-k indices
        k = min(n_peaks, len(flat))
        top_indices = np.argpartition(flat, -k)[-k:]
        top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]

        nf, na = scd_mag.shape
        features = np.zeros(3 * n_peaks, dtype=np.float32)
        max_val = flat.max() if flat.max() > 0 else 1.0
        for i, idx in enumerate(top_indices):
            f_idx, a_idx = divmod(idx, na)
            features[3 * i] = flat[idx] / max_val
            features[3 * i + 1] = f_idx / max(nf - 1, 1)
            features[3 * i + 2] = a_idx / max(na - 1, 1)
        return features

    def extract_features(self, iq: np.ndarray) -> np.ndarray:
        """Extract feature vector from a single IQ signal.

        Args:
            iq: 1-D ``complex64`` or ``complex128`` array.

        Returns:
            1-D ``float32`` feature array.
        """
        parts = []
        if self.feature_set in ("cumulants", "combined"):
            parts.append(self._cumulant_features(iq))
        if self.feature_set in ("cyclic_peaks", "combined"):
            parts.append(self._cyclic_peak_features(iq))
        return np.concatenate(parts).astype(np.float32)

    # ------------------------------------------------------------------
    # Train / predict
    # ------------------------------------------------------------------

    def _build_model(self):
        try:
            if self.classifier_type == "decision_tree":
                from sklearn.tree import DecisionTreeClassifier

                return DecisionTreeClassifier(random_state=42)
            else:
                from sklearn.ensemble import RandomForestClassifier

                return RandomForestClassifier(n_estimators=100, random_state=42)
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for CyclostationaryAMC. "
                "Install it with: pip install 'spectra[classifiers]'"
            ) from exc

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CyclostationaryAMC":
        """Fit the classifier.

        Args:
            X: Feature matrix ``[n_samples, n_features]``.
            y: Label array ``[n_samples]``.

        Returns:
            ``self`` for chaining.
        """
        self._model = self._build_model()
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict modulation labels.

        Args:
            X: Feature matrix ``[n_samples, n_features]``.

        Returns:
            Integer label array ``[n_samples]``.
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._model.predict(X)

    def fit_from_dataset(
        self, dataset: "CyclostationaryDataset", max_samples: Optional[int] = None
    ) -> "CyclostationaryAMC":
        """Convenience: extract features from a dataset and fit.

        This generates IQ samples from the dataset's waveform pool,
        extracts features, and fits the classifier in one call.

        Args:
            dataset: A :class:`CyclostationaryDataset` instance.
            max_samples: Cap the number of samples used (default: all).

        Returns:
            ``self`` for chaining.
        """
        n = len(dataset) if max_samples is None else min(max_samples, len(dataset))

        features_list = []
        labels_list = []
        for i in range(n):
            data_dict, label = dataset[i]
            # Re-generate raw IQ from the dataset's seeding to extract our features
            iq = self._regenerate_iq(dataset, i)
            features_list.append(self.extract_features(iq))
            labels_list.append(label)

        X = np.stack(features_list)
        y = np.array(labels_list)
        return self.fit(X, y)

    @staticmethod
    def _regenerate_iq(dataset: "CyclostationaryDataset", idx: int) -> np.ndarray:
        """Reproduce the raw IQ signal for a given dataset index."""
        rng = np.random.default_rng(seed=(dataset.seed, idx))
        waveform_idx = int(rng.integers(0, len(dataset.waveform_pool)))
        waveform = dataset.waveform_pool[waveform_idx]
        sps = getattr(waveform, "samples_per_symbol", 8)
        num_symbols = dataset.num_iq_samples // sps + 1
        sig_seed = int(rng.integers(0, 2**32))
        iq = waveform.generate(
            num_symbols=num_symbols,
            sample_rate=dataset.sample_rate,
            seed=sig_seed,
        )
        iq = iq[: dataset.num_iq_samples]
        if len(iq) < dataset.num_iq_samples:
            padded = np.zeros(dataset.num_iq_samples, dtype=np.complex64)
            padded[: len(iq)] = iq
            iq = padded
        if dataset.impairments is not None:
            from spectra.scene.signal_desc import SignalDescription

            desc = SignalDescription(
                t_start=0.0,
                t_stop=dataset.num_iq_samples / dataset.sample_rate,
                f_low=-waveform.bandwidth(dataset.sample_rate) / 2,
                f_high=waveform.bandwidth(dataset.sample_rate) / 2,
                label=waveform.label,
                snr=0.0,
            )
            iq, _ = dataset.impairments(iq, desc, sample_rate=dataset.sample_rate)
        return iq
