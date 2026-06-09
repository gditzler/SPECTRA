"""SpectrogramDataset: wraps any IQ Dataset with an STFT→Spectrogram pipeline.

This removes the repetitive boiler-plate of manually wrapping ``NarrowbandDataset``
(or any other dataset that returns raw IQ) with ``STFT`` / ``Spectrogram``.
It guarantees the output shape ``[B, 1, F, T]`` required by 2-D image-based
classifiers.

Optional deterministic memory caching is supported so that the exact same
spectrogram for a given ``(dataset, index)`` pair is not re-computed on every
epoch.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar

import torch
from torch.utils.data import Dataset

T_co = TypeVar("T_co", covariant=True)


class SpectrogramDataset(Dataset[Tuple[torch.Tensor, T_co]], Generic[T_co]):
    """Wrap any Dataset that returns ``(iq_array, target)`` and produce a
    spectrogram in ``[1, F, T]`` format.

    Args:
        dataset: Underlying dataset. Its first return element must be a 1-D
            ``complex64``/``complex128`` NumPy array **or** a ``Tensor[2, N]``.
        transform: Callable that converts IQ to a spectrogram-like Tensor.
            Must return a Tensor of shape ``[C, F, T]`` or ``[F, T]``.
            Typical choices: ``STFT()`` (magnitude) or ``Spectrogram()``.
        cache: If ``"memory"``, pre-computed spectrograms are stored in RAM.
            If ``None`` (default), spectrograms are re-computed every call.
        cache_size: Maximum number of cached items.  ``None`` means unlimited.

    Returns (from ``__getitem__``):
        ``(spectrogram_tensor, target)`` where *spectrogram_tensor* has shape
        ``[1, F, T]`` (single-channel magnitude spectrogram).

    Example::

        from spectra import NarrowbandDataset, QPSK, BPSK, AWGN, Compose
        from spectra.transforms import STFT
        from spectra.datasets import SpectrogramDataset

        iq_ds = NarrowbandDataset(
            waveform_pool=[QPSK(), BPSK()],
            num_samples=1000,
            num_iq_samples=1024,
            sample_rate=1e6,
            impairments=Compose([AWGN(snr_range=(5, 20))]),
            seed=42,
        )
        spec_ds = SpectrogramDataset(iq_ds, transform=STFT(nfft=256), cache="memory")
        spec, label = spec_ds[0]   # spec: [1, 129, 16]
    """

    def __init__(
        self,
        dataset: Dataset[Tuple[Any, T_co]],
        transform: Callable[[Any], torch.Tensor],
        cache: Optional[str] = None,
        cache_size: Optional[int] = None,
    ):
        self.dataset = dataset
        self.transform = transform
        self.cache_mode = cache
        self.cache: Dict[int, torch.Tensor] = {}
        self.cache_size = cache_size
        self._shape: Optional[Tuple[int, int, int]] = None

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, T_co]:
        if self.cache_mode == "memory" and index in self.cache:
            return self.cache[index], self._get_target(index)

        iq, target = self.dataset[index]

        # Accept numpy complex arrays or torch tensors shaped [2, N]
        if isinstance(iq, torch.Tensor):
            # Convert [2, N] back to complex for the transform if needed
            if iq.dim() == 2 and iq.shape[0] == 2:
                iq = iq[0].numpy() + 1j * iq[1].numpy()
            else:
                # Already a complex tensor / spectrogram
                pass

        # Ensure numpy array for the transform (STFT expects np.ndarray)
        if isinstance(iq, torch.Tensor) and not torch.is_complex(iq):
            iq = iq.numpy()
        elif hasattr(iq, "numpy"):
            iq = iq.numpy()

        # Apply the spectrogram transform
        spec = self.transform(iq)  # type: ignore[arg-type]

        # Normalize to [1, F, T]
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)
        if spec.dim() != 3:
            raise RuntimeError(
                f"SpectrogramDataset expected transform output of shape [C, F, T], "
                f"got {spec.shape} (ndim={spec.dim()})."
            )

        self._shape = tuple(spec.shape)

        if self.cache_mode == "memory":
            _trim_cache(self.cache, self.cache_size)
            self.cache[index] = spec.detach().clone()

        return spec, target

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_target(self, index: int) -> T_co:
        """Fetch target without re-running the transform."""
        _, target = self.dataset[index]
        return target

    def clear_cache(self) -> None:
        """Drop all cached spectrograms."""
        self.cache.clear()

    @property
    def freq_bins(self) -> int:
        """Number of frequency bins in the output spectrogram.

        Requires at least one sample to have been generated.
        """
        if self._shape is None:
            raise RuntimeError(
                "freq_bins is only available after at least one __getitem__ call."
            )
        return int(self._shape[1])

    @property
    def time_bins(self) -> int:
        """Number of time bins in the output spectrogram.

        Requires at least one sample to have been generated.
        """
        if self._shape is None:
            raise RuntimeError(
                "time_bins is only available after at least one __getitem__ call."
            )
        return int(self._shape[2])


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _trim_cache(cache: Dict[int, torch.Tensor], max_size: Optional[int]) -> None:
    """Evict oldest entry if cache exceeds max_size."""
    if max_size is None:
        return
    while len(cache) >= max_size:
        # Arbitrary eviction – oldest key
        first_key = next(iter(cache))
        del cache[first_key]
