"""Domain adaptation transforms for cross-source IQ signal alignment.

These transforms normalize IQ signals across different capture sources,
SDR hardware, and gain settings. They inherit from the impairment
Transform ABC and are composable via Compose.

Statistical alignment (Tier 1): DCRemove, ClipNormalize, PowerNormalize,
AGCNormalize, Resample.

Spectral alignment (Tier 2): SpectralWhitening, NoiseFloorMatch,
BandpassAlign.

Reference-based (Tier 3 stubs): NoiseProfileTransfer, ReceiverEQ.
"""

from typing import Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class DCRemove(Transform):
    """Remove DC offset via mean subtraction.

    Many SDR receivers introduce a DC spur at the center frequency.
    This transform removes it by subtracting the mean of the signal.
    """

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        return (iq - np.mean(iq)).astype(np.complex64), desc


class ClipNormalize(Transform):
    """Clip outlier samples beyond N sigma and scale to [-1, 1].

    Useful for taming signals with extreme amplitude spikes from
    ADC saturation or interference before feeding to a model.

    Args:
        clip_sigma: Clip threshold in standard deviations. Default 3.0.
    """

    def __init__(self, clip_sigma: float = 3.0):
        self._clip_sigma = clip_sigma

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        re = iq.real.copy()
        im = iq.imag.copy()

        re_std = np.std(re)
        im_std = np.std(im)

        if re_std > 0:
            thresh_re = self._clip_sigma * re_std
            np.clip(re, -thresh_re, thresh_re, out=re)
        if im_std > 0:
            thresh_im = self._clip_sigma * im_std
            np.clip(im, -thresh_im, thresh_im, out=im)

        peak = max(np.max(np.abs(re)), np.max(np.abs(im)))
        if peak > 0:
            re /= peak
            im /= peak

        return (re + 1j * im).astype(np.complex64), desc


class PowerNormalize(Transform):
    """Scale IQ signal to a target RMS power level in dBFS.

    Useful for normalizing signals captured at different gain settings
    to a consistent power level before training.

    Args:
        target_power_dbfs: Target RMS power in dB relative to full scale.
            Default -20.0.
    """

    def __init__(self, target_power_dbfs: float = -20.0):
        self._target_dbfs = target_power_dbfs

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        rms = np.sqrt(np.mean(np.abs(iq) ** 2))
        if rms == 0:
            return iq, desc
        target_linear = 10.0 ** (self._target_dbfs / 20.0)
        return (iq * (target_linear / rms)).astype(np.complex64), desc


class AGCNormalize(Transform):
    """Normalize gain to undo differences in hardware AGC settings.

    Two modes:
    - ``"rms"``: scale so RMS equals ``target_level`` (unit power default)
    - ``"peak"``: scale so max absolute value equals ``target_level``

    Args:
        method: ``"rms"`` or ``"peak"``. Default ``"rms"``.
        target_level: Target normalization level. Default 1.0.

    Raises:
        ValueError: If method is not ``"rms"`` or ``"peak"``.
    """

    def __init__(self, method: str = "rms", target_level: float = 1.0):
        if method not in ("rms", "peak"):
            raise ValueError(f"method must be 'rms' or 'peak', got '{method}'")
        self._method = method
        self._target = target_level

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        if self._method == "rms":
            level = np.sqrt(np.mean(np.abs(iq) ** 2))
        else:
            level = np.max(np.abs(iq))

        if level == 0:
            return iq, desc
        return (iq * (self._target / level)).astype(np.complex64), desc


class Resample(Transform):
    """Rational resampling to a target sample rate.

    Uses ``scipy.signal.resample_poly`` with rational approximation.
    Requires ``scipy`` — install with ``pip install spectra[alignment]``.

    The ``sample_rate`` keyword argument is required (forwarded by
    ``Compose`` or passed directly).

    Args:
        target_sample_rate: Target sample rate in Hz.

    Raises:
        ImportError: If scipy is not installed.
        ValueError: If ``sample_rate`` kwarg is missing.

    .. note:: The design spec calls for updating ``desc`` sample rate
       metadata. ``SignalDescription`` does not currently carry a
       ``sample_rate`` field; when one is added, this transform should
       be updated to set it on the returned desc.
    """

    def __init__(self, target_sample_rate: float):
        try:
            from scipy.signal import resample_poly  # noqa: F401
        except ImportError:
            raise ImportError(
                "Resample requires scipy. Install with: pip install spectra[alignment]"
            ) from None
        self._target_rate = target_sample_rate

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        from fractions import Fraction

        from scipy.signal import resample_poly

        sample_rate = kwargs.get("sample_rate")
        if sample_rate is None:
            raise ValueError(
                "Resample requires sample_rate kwarg. "
                "Pass via Compose(...)(iq, desc, sample_rate=fs) or directly."
            )

        if sample_rate == self._target_rate:
            return iq, desc

        frac = Fraction(self._target_rate / sample_rate).limit_denominator(1000)
        up, down = frac.numerator, frac.denominator

        resampled = resample_poly(iq, up, down).astype(np.complex64)
        return resampled, desc


class SpectralWhitening(Transform):
    """Flatten PSD by dividing by the smoothed spectral envelope.

    Removes receiver-specific frequency coloring without needing a
    reference measurement. Preserves signal phase.

    Args:
        smoothing_window: Moving average window size for PSD smoothing.
            Larger values produce more aggressive flattening. Default 64.
    """

    def __init__(self, smoothing_window: int = 64):
        self._window = smoothing_window

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        spectrum = np.fft.fft(iq)
        magnitude = np.abs(spectrum)

        kernel = np.ones(self._window) / self._window
        smoothed = np.convolve(magnitude, kernel, mode="same")

        smoothed_max = np.max(smoothed)
        if smoothed_max == 0:
            return iq, desc

        floor = smoothed_max * 1e-10
        smoothed = np.maximum(smoothed, floor)

        whitened_spectrum = spectrum / smoothed

        original_power = np.mean(np.abs(iq) ** 2)
        result = np.fft.ifft(whitened_spectrum)
        result_power = np.mean(np.abs(result) ** 2)
        if result_power > 0:
            result *= np.sqrt(original_power / result_power)

        return result.astype(np.complex64), desc


class NoiseFloorMatch(Transform):
    """Estimate noise floor and scale to match a target level.

    Useful when combining captures with different noise figures
    or receiver sensitivities.

    Args:
        target_noise_floor_db: Target noise floor in dB. Default -40.0.
        estimation_method: ``"median"`` (robust) or ``"minimum"``
            (lower bound). Default ``"median"``.
    """

    def __init__(self, target_noise_floor_db: float = -40.0, estimation_method: str = "median"):
        if estimation_method not in ("median", "minimum"):
            raise ValueError(
                f"estimation_method must be 'median' or 'minimum', got '{estimation_method}'"
            )
        self._target_db = target_noise_floor_db
        self._method = estimation_method

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        spectrum = np.fft.fft(iq)
        psd = np.abs(spectrum) ** 2 / len(iq)
        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-30))

        if self._method == "median":
            floor_db = float(np.median(psd_db))
        else:
            floor_db = float(np.min(psd_db))

        scale = 10.0 ** ((self._target_db - floor_db) / 20.0)
        return (iq * scale).astype(np.complex64), desc


class BandpassAlign(Transform):
    """Frequency-shift and bandpass-filter a signal to a target center and bandwidth.

    Estimates the signal's center of mass in frequency, shifts to align it
    with ``center_freq``, then applies a rectangular bandpass filter in the
    frequency domain. Updates ``desc.f_low`` and ``desc.f_high``.

    The ``sample_rate`` keyword argument is required.

    Args:
        center_freq: Target center frequency in Hz (relative to baseband).
            Default 0.0.
        bandwidth: Target bandwidth as a fraction of sample rate (0, 1].

    Raises:
        ValueError: If ``sample_rate`` kwarg is missing.
    """

    def __init__(self, center_freq: float = 0.0, bandwidth: float = 0.5):
        self._center = center_freq
        self._bw_frac = bandwidth

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        from dataclasses import replace

        sample_rate = kwargs.get("sample_rate")
        if sample_rate is None:
            raise ValueError(
                "BandpassAlign requires sample_rate kwarg. "
                "Pass via Compose(...)(iq, desc, sample_rate=fs) or directly."
            )
        n = len(iq)

        # Estimate current center of mass in frequency domain
        spectrum = np.fft.fft(iq)
        freqs = np.fft.fftfreq(n, d=1.0 / sample_rate)
        power = np.abs(spectrum) ** 2
        total_power = np.sum(power)
        if total_power > 0:
            current_center = np.sum(freqs * power) / total_power
        else:
            current_center = 0.0

        # Frequency-shift to align center of mass to target
        shift = self._center - current_center
        if shift != 0:
            t_arr = np.arange(n) / sample_rate
            iq = iq * np.exp(1j * 2 * np.pi * shift * t_arr).astype(np.complex64)
            spectrum = np.fft.fft(iq)

        # Bandpass filter
        half_bw = self._bw_frac * sample_rate / 2.0
        mask = np.abs(freqs - self._center) <= half_bw
        spectrum *= mask

        result = np.fft.ifft(spectrum).astype(np.complex64)

        new_desc = replace(
            desc,
            f_low=self._center - half_bw,
            f_high=self._center + half_bw,
        )
        return result, new_desc


class NoiseProfileTransfer(Transform):
    """Replace synthetic noise with noise characteristics from a real capture.

    .. note:: This is a research-grade transform. It is not yet implemented
       and will raise ``NotImplementedError`` when called.

    Intended approach:

    1. Estimate and subtract signal component from ``noise_source``
       (or use a known noise-only capture).
    2. Estimate noise PSD profile from the reference.
    3. Generate colored noise matching the reference PSD.
    4. Replace the synthetic noise component in the input signal.

    Args:
        noise_source: Path to noise capture file, or raw complex64 noise
            array from a real receiver.
    """

    def __init__(self, noise_source):
        self._source = noise_source

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        raise NotImplementedError(
            "NoiseProfileTransfer is a planned research transform. "
            "Contributions welcome — see the design spec in "
            "docs/plans/2026-03-11-domain-adaptation-transforms.md"
        )


class ReceiverEQ(Transform):
    """Equalize receiver frequency response using a reference PSD profile.

    .. note:: This is a research-grade transform. It is not yet implemented
       and will raise ``NotImplementedError`` when called.

    Intended approach:

    1. Load or accept a reference PSD from a calibration capture of a
       known flat-spectrum signal.
    2. Compute the ratio ``current_psd / reference_psd``.
    3. Apply the inverse filter to equalize the receiver response.

    Args:
        reference_psd: Reference PSD array or path to file containing
            one. Should be from a flat-spectrum calibration signal.
    """

    def __init__(self, reference_psd):
        self._ref = reference_psd

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        raise NotImplementedError(
            "ReceiverEQ is a planned research transform. "
            "Contributions welcome — see the design spec in "
            "docs/plans/2026-03-11-domain-adaptation-transforms.md"
        )
