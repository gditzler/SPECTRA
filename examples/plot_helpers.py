"""Shared plotting helpers for SPECTRA examples."""

import os

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def savefig(name: str, dpi: int = 150) -> None:
    """Save current figure to outputs/ directory."""
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")


def plot_iq_time(iq: np.ndarray, title: str = "", num_samples: int = 200) -> None:
    """Plot I and Q components vs sample index."""
    n = min(num_samples, len(iq))
    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    axes[0].plot(iq[:n].real, linewidth=0.8)
    axes[0].set_ylabel("In-Phase (I)")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(iq[:n].imag, linewidth=0.8, color="tab:orange")
    axes[1].set_ylabel("Quadrature (Q)")
    axes[1].set_xlabel("Sample Index")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()


def plot_constellation(iq: np.ndarray, title: str = "", max_pts: int = 2000) -> None:
    """Scatter plot of IQ constellation."""
    pts = iq[: min(max_pts, len(iq))]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pts.real, pts.imag, s=2, alpha=0.5)
    ax.set_xlabel("In-Phase")
    ax.set_ylabel("Quadrature")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def plot_psd(
    iq: np.ndarray, sample_rate: float, title: str = "", nfft: int = 1024
) -> None:
    """Plot power spectral density."""
    fig, ax = plt.subplots(figsize=(10, 4))
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
    spectrum = np.fft.fftshift(np.fft.fft(iq[:nfft], n=nfft))
    psd = 10 * np.log10(np.abs(spectrum) ** 2 + 1e-12)
    ax.plot(freqs / 1e3, psd, linewidth=0.8)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def plot_spectrogram_img(
    spec: np.ndarray, title: str = "", cmap: str = "viridis"
) -> None:
    """Plot a 2D spectrogram array (freq x time)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
    )
    ax.set_xlabel("Time Bin")
    ax.set_ylabel("Frequency Bin")
    ax.set_title(title)
    fig.tight_layout()
