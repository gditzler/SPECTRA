"""DatasetProfiler: lightweight inspection for generated SPECTRA datasets.

Quickly sample an on-the-fly dataset and report per-class counts, SNR
histograms, mean IQ power, and spectrogram statistics — all without writing
a custom analysis script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import numpy as np
import torch
from torch.utils.data import Dataset

_T = TypeVar("_T")


# ---------------------------------------------------------------------------
# Helper types
# ---------------------------------------------------------------------------


Numeric = float | int | np.ndarray | torch.Tensor


def _to_float(val: Numeric) -> float:
    """Convert scalar numeric to Python float."""
    if isinstance(val, torch.Tensor):
        return float(val.detach().cpu().item())
    if isinstance(val, np.ndarray):
        return float(val.item())
    return float(val)


def _extract_snr(desc: Any) -> float | None:
    """Try to extract SNR (dB) from a SignalDescription or similar object."""
    # Standard SignalDescription path
    if hasattr(desc, "snr"):
        try:
            return _to_float(desc.snr)
        except Exception:
            pass
    # Fallback: dict-like path
    if isinstance(desc, dict) and "snr" in desc:
        try:
            return _to_float(desc["snr"])
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Profile result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DatasetProfile:
    """Container for profiling statistics.

    Attributes:
        num_samples: Total number of samples inspected.
        class_counts: Mapping from class label (string) to sample count.
        snr_histogram: Dict with bins (List[float]) and counts (List[int]).
        average_power_db: Average signal power across inspected samples.
        spectrogram_mean: Optional mean of spectrogram magnitudes (if applicable).
        spectrogram_std: Optional std of spectrogram magnitudes (if applicable).
    """

    num_samples: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    snr_histogram: Dict[str, Any] = field(default_factory=dict)
    average_power_db: float = 0.0
    spectrogram_mean: Optional[float] = None
    spectrogram_std: Optional[float] = None

    def summary(self) -> str:
        """One-line human-readable summary."""
        total = self.num_samples
        counts = self.class_counts
        snr = self.snr_histogram.get("counts", [])
        if snr:
            total_snr = sum(snr)
        else:
            total_snr = 0
        return (
            f"DatasetProfile(samples={total}, classes={len(counts)}, "
            f"avg_pwr={self.average_power_db:.2f} dB, "
            f"snr_bins={total_snr})"
        )

    def __repr__(self) -> str:
        return f"DatasetProfile({self.summary()})"


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


class DatasetProfiler:
    """Light-weight profiler that samples a SPECTRA dataset and reports
    aggregate statistics.

    Args:
        dataset: Any PyTorch ``Dataset`` returning ``(data, label)``.
        label_extractor: Optional callable that extracts a string label
            from the dataset's ``label`` return value.  Defaults to
            ``str(label)``.
        max_samples: Number of samples to inspect.  ``None`` inspects the
            entire dataset (can be slow for large synthetic sets).

    Example::

        from spectra import NarrowbandDataset, QPSK, BPSK, AWGN, Compose
        from spectra.datasets import DatasetProfiler

        ds = NarrowbandDataset(
            waveform_pool=[QPSK(), BPSK()],
            num_samples=1000,
            num_iq_samples=1024,
            sample_rate=1e6,
            impairments=Compose([AWGN(snr_range=(5, 20))]),
            seed=42,
        )
        profiler = DatasetProfiler(ds, max_samples=256)
        profile = profiler.run()
        print(profile.summary())
    """

    def __init__(
        self,
        dataset: Dataset[Tuple[Any, _T]],
        label_extractor: Optional[Callable[[_T], str]] = None,
        max_samples: Optional[int] = None,
    ):
        self.dataset = dataset
        self.label_extractor = label_extractor or (lambda x: str(x))
        self.max_samples = max_samples

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> DatasetProfile:
        """Profile the dataset and return a :class:`DatasetProfile`."""
        n = len(self.dataset)  # type: ignore[arg-type]
        max_s = self.max_samples if self.max_samples is not None else n
        max_s = min(max_s, n)

        class_counts: Dict[str, int] = {}
        snr_values: List[float] = []
        power_db_values: List[float] = []
        spec_values: List[np.ndarray] = []

        for idx in range(max_s):
            item = self.dataset[idx]
            data, *rest = item
            label = item[1] if len(item) > 1 else None
            label_str = self.label_extractor(label)
            class_counts[label_str] = class_counts.get(label_str, 0) + 1
            if isinstance(item, tuple) and len(item) >= 3:
                last = item[-1]
                if isinstance(last, (int, float, np.floating)):
                    try:
                        snr_values.append(float(last))
                    except Exception:
                        pass

            # --- IQ / magnitude data analysis --------------------------------
            spec = self._unwrap_data(data)
            if spec is None:
                continue

            # Average power (dB) over the whole sample
            avg_pwr = self._compute_power(spec)
            if avg_pwr is not None:
                power_db_values.append(avg_pwr)

            # Try to collect spectrogram stats
            if isinstance(spec, torch.Tensor) and spec.dim() >= 2:
                spec_np = spec.detach().cpu().float().numpy()
                spec_values.append(spec_np)

        # --- Aggregate ---------------------------------------------------
        profile = DatasetProfile(
            num_samples=max_s,
            class_counts=class_counts,
            average_power_db=float(np.mean(power_db_values)) if power_db_values else 0.0,
        )

        # SNR histogram
        if snr_values:
            bins, counts = self._histogram(snr_values)
            profile.snr_histogram = {"bins": bins, "counts": counts}

        # Spectrogram stats
        if spec_values:
            all_specs = np.concatenate([s.ravel() for s in spec_values])
            profile.spectrogram_mean = float(np.mean(all_specs))
            profile.spectrogram_std = float(np.std(all_specs))

        return profile

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_data(data: Any) -> Optional[np.ndarray | torch.Tensor]:
        """Normalise dataset returns into an array/tensor we can inspect."""
        # If the dataset returns a dict (e.g., wideband COCO), ignore for now.
        if isinstance(data, dict):
            return None
        if isinstance(data, (torch.Tensor, np.ndarray)):
            return data
        return None

    @staticmethod
    def _compute_power(data: np.ndarray | torch.Tensor) -> Optional[float]:
        """Compute average signal power in dB."""
        try:
            if isinstance(data, torch.Tensor):
                # Data could be [2, N] (I/Q) or [C, F, T]
                if data.dim() == 2 and data.shape[0] == 2:
                    # Raw IQ: reconstruct complex
                    iq = data[0] + 1j * data[1]
                    power = torch.mean(torch.abs(iq) ** 2).item()
                else:
                    power = torch.mean(data ** 2).item()
                if power <= 0:
                    return -np.inf
                return float(10.0 * np.log10(power))
            else:
                # NumPy
                if data.dtype == np.complex64 or data.dtype == np.complex128:
                    power = np.mean(np.abs(data) ** 2)
                else:
                    power = np.mean(data ** 2)
                if power <= 0:
                    return -np.inf
                return float(10.0 * np.log10(power))
        except Exception:
            return None

    @staticmethod
    def _histogram(values: List[float], num_bins: int = 10) -> Tuple[List[float], List[int]]:
        """Simple histogram of scalar values."""
        arr = np.array(values)
        lo, hi = float(arr.min()), float(arr.max())
        if lo == hi:
            return [lo], [len(values)]
        edges = np.linspace(lo, hi, num_bins + 1)
        counts, _ = np.histogram(arr, bins=edges)
        # Return bin centres
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(num_bins)]
        return centers, counts.tolist()

    # ------------------------------------------------------------------
    # Convenience wrappers for known dataset types
    # ------------------------------------------------------------------

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset[Tuple[Any, _T]],
        max_samples: Optional[int] = None,
    ) -> "DatasetProfiler":
        """Create a profiler with an appropriate label extractor preset."""
        preset = _detect_preset(dataset)
        if preset == "narrowband":
            return cls(dataset, label_extractor=_narrowband_label, max_samples=max_samples)
        if preset == "wideband":
            return cls(dataset, label_extractor=_wideband_label, max_samples=max_samples)
        if preset == "radar":
            return cls(dataset, label_extractor=_radar_label, max_samples=max_samples)
        return cls(dataset, max_samples=max_samples)


# ---------------------------------------------------------------------------
# Label extractor presets
# ---------------------------------------------------------------------------


_NARROWBAND_TYPES = {"NarrowbandDataset", "CyclostationaryDataset", "SNRSweepDataset"}
_WIDEBAND_TYPES = {"WidebandDataset", "WidebandDirectionFindingDataset"}
_RADAR_TYPES = {"RadarDataset", "RadarPipelineDataset"}


def _detect_preset(dataset: Any) -> str:
    cls_name = type(dataset).__name__
    if cls_name in _NARROWBAND_TYPES:
        return "narrowband"
    if cls_name in _WIDEBAND_TYPES:
        return "wideband"
    if cls_name in _RADAR_TYPES:
        return "radar"
    return "generic"


def _narrowband_label(label: Any) -> str:
    # NarrowbandDataset returns int class index
    if isinstance(label, int):
        return f"class_{label}"
    return str(label)


def _wideband_label(target: Any) -> str:
    # WidebandDataset returns COCO-style dict
    if isinstance(target, dict):
        n = len(target.get("labels", []))
        return f"{n}_signals"
    return str(target)


def _radar_label(target: Any) -> str:
    # RadarDataset returns RadarTarget
    if hasattr(target, "num_targets"):
        return f"n_targets_{target.num_targets}"
    return str(target)
