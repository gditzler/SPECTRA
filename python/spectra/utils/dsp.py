from typing import Optional

import numpy as np

from spectra._rust import (
    apply_rrc_filter as _rrc_filter,
)
from spectra._rust import (
    convolve_complex as _convolve_complex,
)
from spectra._rust import (
    gaussian_taps as _gaussian_taps,
)
from spectra._rust import (
    lowpass_taps as _lowpass_taps,
)

_RESAMPLE_TAPS_MULTIPLIER = 64  # taps-per-rate for polyphase anti-aliasing filter


def low_pass(num_taps: int, cutoff: float) -> np.ndarray:
    """Windowed-sinc (Blackman) FIR lowpass filter taps.

    Args:
        num_taps: Number of filter taps.
        cutoff: Normalized cutoff frequency in (0, 1) where 1 = Nyquist.
    """
    return np.array(_lowpass_taps(num_taps, cutoff), dtype=np.float32)


def srrc_taps(num_symbols: int, rolloff: float, sps: int) -> np.ndarray:
    """Extract SRRC filter taps by filtering an impulse.

    Args:
        num_symbols: Filter span in symbols.
        rolloff: Roll-off factor.
        sps: Samples per symbol.
    """
    # Create single impulse symbol and filter it
    impulse = np.array([1.0 + 0j], dtype=np.complex64)
    filtered = np.array(_rrc_filter(impulse, rolloff, num_symbols, sps))
    return filtered.real.astype(np.float32)


def gaussian_taps(bt: float, span: int, sps: int) -> np.ndarray:
    """Gaussian filter taps (for GFSK/GMSK).

    Args:
        bt: Bandwidth-time product.
        span: Filter span in symbols.
        sps: Samples per symbol.
    """
    return np.array(_gaussian_taps(bt, span, sps), dtype=np.float32)


def frequency_shift(iq: np.ndarray, offset: float, sample_rate: float) -> np.ndarray:
    """Apply frequency shift to IQ signal."""
    t = np.arange(len(iq)) / sample_rate
    shift = np.exp(1j * 2.0 * np.pi * offset * t).astype(np.complex64)
    return (iq * shift).astype(np.complex64)


def upsample(signal: np.ndarray, factor: int) -> np.ndarray:
    """Zero-insertion upsampling."""
    out = np.zeros(len(signal) * factor, dtype=signal.dtype)
    out[::factor] = signal
    return out


def convolve(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Convolve complex signal with real filter taps (via Rust)."""
    sig = signal.astype(np.complex64)
    t = taps.astype(np.float32)
    return np.array(_convolve_complex(sig, t))


def polyphase_interpolator(signal: np.ndarray, taps: np.ndarray, factor: int) -> np.ndarray:
    """Polyphase upsampling (interpolation)."""
    # Pad taps to multiple of factor
    n_taps = len(taps)
    pad = (factor - n_taps % factor) % factor
    taps_padded = np.concatenate([taps, np.zeros(pad, dtype=taps.dtype)])
    # Reshape into polyphase branches
    branches = taps_padded.reshape(factor, -1)
    # Upsample signal
    upsampled = upsample(signal, factor)
    out = np.zeros_like(upsampled)
    for i in range(factor):
        branch_out = np.convolve(signal, branches[i], mode="full")[: len(signal)]
        out[i::factor] = branch_out
    return out.astype(signal.dtype)


def polyphase_decimator(signal: np.ndarray, taps: np.ndarray, factor: int) -> np.ndarray:
    """Polyphase downsampling (decimation)."""
    # Filter then decimate
    filtered = convolve(signal, taps)
    # Take every factor-th sample, center crop
    start = (len(filtered) - len(signal)) // 2
    filtered_cropped = filtered[start : start + len(signal)]
    return filtered_cropped[::factor]


def multistage_resampler(signal: np.ndarray, up: int, down: int) -> np.ndarray:
    """Rational resampling: upsample by up, filter, downsample by down."""
    # Design anti-aliasing filter
    cutoff = min(1.0 / up, 1.0 / down)
    taps = low_pass(_RESAMPLE_TAPS_MULTIPLIER * max(up, down) + 1, cutoff)
    # Upsample
    upsampled = upsample(signal, up)
    # Filter
    filtered = convolve(upsampled, taps * up)
    # Downsample
    start = (len(filtered) - len(upsampled)) // 2
    filtered_cropped = filtered[start : start + len(upsampled)]
    return filtered_cropped[::down]


def noise_generator(
    num_samples: int,
    power: float = 1.0,
    color: str = "white",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate complex noise with specified color and power.

    Args:
        num_samples: Number of samples.
        power: Target noise power.
        color: 'white', 'pink', or 'red'.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    noise = (rng.standard_normal(num_samples) + 1j * rng.standard_normal(num_samples)).astype(
        np.complex64
    ) / np.sqrt(2.0)

    if color != "white":
        freqs = np.fft.rfftfreq(num_samples)
        freqs[0] = 1.0  # avoid div by zero
        if color == "pink":
            spectrum = 1.0 / np.sqrt(freqs)
        elif color == "red":
            spectrum = 1.0 / freqs
        else:
            spectrum = np.ones_like(freqs)
        # Shape in frequency domain
        X = np.fft.rfft(noise.real)
        noise_real = np.fft.irfft(X * spectrum, n=num_samples)
        X = np.fft.rfft(noise.imag)
        noise_imag = np.fft.irfft(X * spectrum, n=num_samples)
        noise = (noise_real + 1j * noise_imag).astype(np.complex64)

    # Scale to target power
    current_power = np.mean(np.abs(noise) ** 2)
    if current_power > 0:
        noise *= np.sqrt(power / current_power)
    return noise


def compute_spectrogram(iq: np.ndarray, nfft: int = 256, hop: int = 64) -> np.ndarray:
    """Compute magnitude spectrogram.

    Returns 2D real array of shape [nfft, num_frames].
    """
    window = np.hanning(nfft)
    num_frames = (len(iq) - nfft) // hop + 1
    spec = np.zeros((nfft, max(num_frames, 0)), dtype=np.float32)
    for i in range(num_frames):
        start = i * hop
        segment = iq[start : start + nfft] * window
        X = np.fft.fftshift(np.fft.fft(segment))
        spec[:, i] = np.abs(X).astype(np.float32)
    return spec


def center_freq_from_bounds(f_low: float, f_high: float) -> float:
    """Compute center frequency from lower and upper bounds."""
    return (f_low + f_high) / 2.0


def bandwidth_from_bounds(f_low: float, f_high: float) -> float:
    """Compute bandwidth from lower and upper bounds."""
    return f_high - f_low
