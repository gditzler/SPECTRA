# tests/test_studio_plotting.py
"""Tests for SPECTRA Studio plotting functions."""
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI


def _make_iq(n=1024, seed=42):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


def test_plot_iq_returns_figure():
    from spectra.studio.plotting import plot_iq
    fig = plot_iq(_make_iq(), sample_rate=1e6)
    assert fig is not None
    assert hasattr(fig, "savefig")  # matplotlib Figure


def test_plot_fft_returns_figure():
    from spectra.studio.plotting import plot_fft
    fig = plot_fft(_make_iq(), sample_rate=1e6)
    assert fig is not None


def test_plot_waterfall_returns_figure():
    from spectra.studio.plotting import plot_waterfall
    fig = plot_waterfall(_make_iq(4096), sample_rate=1e6)
    assert fig is not None


def test_plot_constellation_returns_figure():
    from spectra.studio.plotting import plot_constellation
    fig = plot_constellation(_make_iq())
    assert fig is not None


def test_plot_scd_returns_figure():
    from spectra.studio.plotting import plot_scd
    fig = plot_scd(_make_iq(2048), sample_rate=1e6)
    assert fig is not None


def test_plot_ambiguity_returns_figure():
    from spectra.studio.plotting import plot_ambiguity
    fig = plot_ambiguity(_make_iq(512))
    assert fig is not None


def test_plot_eye_returns_figure():
    from spectra.studio.plotting import plot_eye
    fig = plot_eye(_make_iq(1024), samples_per_symbol=8)
    assert fig is not None


def test_plot_eye_incompatible_returns_none():
    from spectra.studio.plotting import plot_eye
    # samples_per_symbol=0 or None should return None (incompatible)
    result = plot_eye(_make_iq(), samples_per_symbol=0)
    assert result is None
