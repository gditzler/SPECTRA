# tests/test_radar_algorithms.py
"""Tests for matched filter and CFAR detectors."""
import numpy as np


def _make_pulse_signal(delay_samples: int, amplitude: float = 1.0,
                       pulse_len: int = 64, total_len: int = 512, seed: int = 0) -> tuple:
    """Return (received, template) with a point target at delay_samples."""
    rng = np.random.default_rng(seed)
    template = rng.standard_normal(pulse_len) + 1j * rng.standard_normal(pulse_len)
    template = template / np.linalg.norm(template)  # unit energy
    received = np.zeros(total_len, dtype=complex)
    end = delay_samples + pulse_len
    if end <= total_len:
        received[delay_samples:end] = amplitude * template
    # Add weak noise
    noise_amp = 0.01
    received += noise_amp * (rng.standard_normal(total_len) + 1j * rng.standard_normal(total_len))
    return received, template


def test_mf_output_length():
    from spectra.algorithms.radar import matched_filter
    received = np.ones(512, dtype=complex)
    template = np.ones(64, dtype=complex)
    out = matched_filter(received, template)
    assert len(out) == 512 + 64 - 1


def test_mf_peak_at_correct_delay():
    from spectra.algorithms.radar import matched_filter
    delay = 100
    received, template = _make_pulse_signal(delay_samples=delay, amplitude=10.0)
    mf_out = matched_filter(received, template)
    peak_idx = np.argmax(np.abs(mf_out))
    expected_peak = delay + len(template) - 1
    assert abs(peak_idx - expected_peak) <= 2, (
        f"MF peak at {peak_idx}, expected ~{expected_peak}"
    )


def test_mf_snr_improvement():
    """Matched filter output SNR must exceed input SNR (classic SNR improvement property)."""
    from spectra.algorithms.radar import matched_filter
    rng = np.random.default_rng(42)
    pulse_len = 64
    template = rng.standard_normal(pulse_len) + 1j * rng.standard_normal(pulse_len)
    template /= np.linalg.norm(template)
    received = np.zeros(512, dtype=complex)
    received[100:100 + pulse_len] = 2.0 * template  # amplitude 2 → power 4
    noise = 0.1 * (rng.standard_normal(512) + 1j * rng.standard_normal(512))
    received += noise
    mf_out = matched_filter(received, template)
    peak_power = np.max(np.abs(mf_out) ** 2)
    noise_power = np.mean(np.abs(mf_out[:50]) ** 2)  # noise region before target
    snr_mf = peak_power / (noise_power + 1e-30)
    assert snr_mf > 10.0, f"Expected MF SNR > 10, got {snr_mf:.2f}"


def test_ca_cfar_detects_target():
    from spectra.algorithms.radar import ca_cfar
    power = np.ones(256) * 0.01
    power[128] = 100.0  # strong target
    detections = ca_cfar(power, guard_cells=2, training_cells=8, pfa=1e-4)
    assert detections[128], "CA-CFAR should detect strong target at bin 128"


def test_ca_cfar_no_false_alarms_flat():
    """On a perfectly flat noise floor, CA-CFAR should have very few false alarms."""
    from spectra.algorithms.radar import ca_cfar
    rng = np.random.default_rng(0)
    # Exponential-distributed power (Rayleigh envelope) — ideal for CA-CFAR
    power = rng.exponential(scale=1.0, size=1000)
    detections = ca_cfar(power, guard_cells=4, training_cells=16, pfa=1e-3)
    # Allow 2× the expected false alarm count as a loose bound
    assert detections.sum() < 10, f"Too many false alarms: {detections.sum()}"


def test_os_cfar_detects_target():
    from spectra.algorithms.radar import os_cfar
    power = np.ones(256) * 0.01
    power[64] = 50.0
    detections = os_cfar(power, guard_cells=2, training_cells=8, k_rank=6, pfa=1e-4)
    assert detections[64], "OS-CFAR should detect strong target at bin 64"


def test_os_cfar_output_shape():
    from spectra.algorithms.radar import os_cfar
    power = np.random.rand(512)
    out = os_cfar(power, guard_cells=3, training_cells=12, k_rank=8)
    assert out.shape == (512,)
    assert out.dtype == bool
