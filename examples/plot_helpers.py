"""Shared plotting helpers for SPECTRA examples."""

import os
from typing import Optional

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


def plot_music_spectrum(
    scan_deg: np.ndarray,
    spectrum: np.ndarray,
    true_azimuths_deg: Optional[np.ndarray] = None,
    estimated_azimuths_deg: Optional[np.ndarray] = None,
    title: str = "MUSIC Pseudospectrum",
) -> None:
    """Plot a MUSIC pseudospectrum with optional true/estimated angle markers.

    Args:
        scan_deg: Scan angles in degrees.
        spectrum: Pseudospectrum values (raw, will be normalised to dB).
        true_azimuths_deg: True source azimuths in degrees (red dashed lines).
        estimated_azimuths_deg: Estimated azimuths in degrees (orange dotted lines).
        title: Figure title.
    """
    spectrum_db = 10 * np.log10(spectrum / spectrum.max() + 1e-30)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(scan_deg, spectrum_db, color="steelblue", linewidth=1.2)
    if true_azimuths_deg is not None:
        for az in true_azimuths_deg:
            ax.axvline(az, color="crimson", linestyle="--", linewidth=1.5, label=f"True {az:.1f}°")
    if estimated_azimuths_deg is not None:
        for az in estimated_azimuths_deg:
            ax.axvline(az, color="orange", linestyle=":", linewidth=1.5, label=f"Est. {az:.1f}°")
    ax.set_xlabel("Azimuth (degrees)")
    ax.set_ylabel("Pseudospectrum (dB, normalised)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def plot_array_geometry(
    positions: np.ndarray,
    title: str = "Array Geometry",
) -> None:
    """Scatter-plot array element positions in the x-y plane.

    Args:
        positions: Element positions in wavelengths, shape (N, 2).
        title: Figure title.
    """
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.scatter(positions[:, 0], positions[:, 1], s=120, zorder=3)
    for i, (x, y) in enumerate(positions):
        ax.annotate(
            f"{i}", (x, y), textcoords="offset points",
            xytext=(0, 8), ha="center", fontsize=8,
        )
    ax.set_xlabel("x (wavelengths)")
    ax.set_title(title)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_rmse_vs_snr(
    snr_rmse: dict,
    label: str = "",
    ax=None,
) -> None:
    """Line plot of DoA RMSE (degrees) vs SNR (dB).

    Args:
        snr_rmse: Dict mapping SNR (dB) → RMSE (degrees), as returned by
            :func:`spectra.metrics.per_snr_rmse`.
        label: Series label for the legend.
        ax: Existing Axes to plot onto; creates a new figure if None.
    """
    snrs = sorted(snr_rmse.keys())
    rmses = [snr_rmse[s] for s in snrs]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("RMSE (degrees)")
        ax.set_title("DoA RMSE vs SNR")
        ax.grid(True, alpha=0.3)
    ax.plot(snrs, rmses, marker="o", linewidth=1.5, label=label or None)
    if label:
        ax.legend(fontsize=9)
    plt.tight_layout()
