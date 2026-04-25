# python/spectra/studio/plotting.py
"""Plot functions for SPECTRA Studio.

All functions accept IQ data (complex64 ndarray) and return a matplotlib
Figure. They are Gradio-agnostic and independently testable.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _dark_style() -> dict:
    """Base style dict for dark-themed plots."""
    return {
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "text.color": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "grid.alpha": 0.5,
    }


def plot_iq(
    iq: np.ndarray,
    sample_rate: float = 1e6,
    start: int = 0,
    num_samples: int = 500,
    dark: bool = True,
) -> plt.Figure:
    """Plot IQ time-domain (I and Q channels vs sample index)."""
    with plt.rc_context(_dark_style() if dark else {}):
        fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
        seg = iq[start : start + num_samples]
        t = np.arange(len(seg)) / sample_rate * 1e6  # microseconds
        axes[0].plot(t, seg.real, color="#4a90d9", linewidth=0.8)
        axes[0].set_ylabel("I")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(t, seg.imag, color="#50b87a", linewidth=0.8)
        axes[1].set_ylabel("Q")
        axes[1].set_xlabel("Time (us)")
        axes[1].grid(True, alpha=0.3)
        fig.suptitle("IQ Time Domain", fontsize=11)
        fig.tight_layout()
    return fig


def plot_fft(
    iq: np.ndarray,
    sample_rate: float = 1e6,
    nfft: int = 1024,
    dark: bool = True,
) -> plt.Figure:
    """Plot FFT / power spectral density (Welch-style)."""
    with plt.rc_context(_dark_style() if dark else {}):
        fig, ax = plt.subplots(figsize=(8, 4))
        from scipy.signal import welch

        freqs, psd = welch(iq, fs=sample_rate, nperseg=min(nfft, len(iq)),
                           return_onesided=False)
        freqs = np.fft.fftshift(freqs)
        psd = np.fft.fftshift(psd)
        ax.plot(freqs / 1e3, 10 * np.log10(psd + 1e-30), color="#4a90d9", linewidth=0.8)
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("PSD (dB/Hz)")
        ax.set_title("Power Spectral Density")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
    return fig


def plot_waterfall(
    iq: np.ndarray,
    sample_rate: float = 1e6,
    nfft: int = 256,
    hop: int = 64,
    dark: bool = True,
) -> plt.Figure:
    """Plot waterfall / spectrogram."""
    with plt.rc_context(_dark_style() if dark else {}):
        fig, ax = plt.subplots(figsize=(8, 5))
        n_frames = max(1, (len(iq) - nfft) // hop)
        spec = np.zeros((nfft, n_frames), dtype=complex)
        for i in range(n_frames):
            seg = iq[i * hop : i * hop + nfft]
            spec[:, i] = np.fft.fftshift(np.fft.fft(seg * np.hanning(nfft)))
        spec_db = 10 * np.log10(np.abs(spec) ** 2 + 1e-30)
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
        # Transpose so time is on Y, frequency on X (per spec)
        ax.imshow(spec_db.T, aspect="auto", origin="lower", cmap="viridis",
                  extent=(float(freqs[0]) / 1e3, float(freqs[-1]) / 1e3, 0.0, float(n_frames)))
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("Time (frame)")
        ax.set_title("Spectrogram")
        fig.tight_layout()
    return fig


def plot_constellation(
    iq: np.ndarray,
    max_points: int = 5000,
    dark: bool = True,
) -> plt.Figure:
    """Plot IQ constellation diagram."""
    with plt.rc_context(_dark_style() if dark else {}):
        fig, ax = plt.subplots(figsize=(5, 5))
        pts = iq[:max_points]
        ax.scatter(pts.real, pts.imag, s=2, alpha=0.5, color="#4a90d9")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.set_title("Constellation")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
    return fig


def plot_scd(
    iq: np.ndarray,
    sample_rate: float = 1e6,
    dark: bool = True,
) -> plt.Figure:
    """Plot Spectral Correlation Density."""
    from spectra.transforms import SCD
    with plt.rc_context(_dark_style() if dark else {}):
        fig, ax = plt.subplots(figsize=(8, 5))
        scd_transform = SCD(nfft=128)
        scd_out = scd_transform(iq)
        if hasattr(scd_out, "numpy"):
            scd_out = scd_out.numpy()
        scd_out = np.squeeze(scd_out)
        ax.imshow(np.abs(scd_out), aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("Spectral Frequency")
        ax.set_ylabel("Cyclic Frequency")
        ax.set_title("Spectral Correlation Density")
        fig.tight_layout()
    return fig


def plot_ambiguity(
    iq: np.ndarray,
    dark: bool = True,
) -> plt.Figure:
    """Plot ambiguity function surface."""
    from spectra.transforms import AmbiguityFunction
    with plt.rc_context(_dark_style() if dark else {}):
        fig, ax = plt.subplots(figsize=(8, 5))
        af = AmbiguityFunction()
        af_out = af(iq)
        if hasattr(af_out, "numpy"):
            af_out = af_out.numpy()
        af_out = np.squeeze(af_out)
        ax.imshow(np.abs(af_out), aspect="auto", origin="lower", cmap="inferno")
        ax.set_xlabel("Delay")
        ax.set_ylabel("Doppler")
        ax.set_title("Ambiguity Function")
        fig.tight_layout()
    return fig


def plot_eye(
    iq: np.ndarray,
    samples_per_symbol: int = 8,
    num_traces: int = 100,
    dark: bool = True,
) -> Optional[plt.Figure]:
    """Plot eye diagram. Returns None if samples_per_symbol is invalid."""
    if samples_per_symbol < 2:
        return None
    sps = samples_per_symbol
    trace_len = 2 * sps  # two symbol periods
    n_available = len(iq) // sps
    n_traces = min(num_traces, max(1, n_available - 2))

    with plt.rc_context(_dark_style() if dark else {}):
        fig, ax = plt.subplots(figsize=(6, 4))
        t = np.arange(trace_len)
        for i in range(n_traces):
            start = i * sps
            if start + trace_len > len(iq):
                break
            ax.plot(t, iq[start : start + trace_len].real, color="#4a90d9",
                    alpha=0.15, linewidth=0.5)
        ax.set_xlabel("Sample (within 2 symbol periods)")
        ax.set_ylabel("Amplitude (I)")
        ax.set_title("Eye Diagram")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
    return fig
