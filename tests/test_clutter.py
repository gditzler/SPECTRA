"""Tests for RadarClutter impairment."""
import numpy as np


def _make_pulse_matrix(num_pulses=32, num_range_bins=128, rng=None):
    """Create a clean pulse matrix for testing."""
    if rng is None:
        rng = np.random.default_rng(42)
    return np.zeros((num_pulses, num_range_bins), dtype=complex)


def test_clutter_output_shape():
    from spectra.impairments.clutter import RadarClutter
    clutter = RadarClutter(cnr=20.0, doppler_spread=50.0, sample_rate=1e6)
    rng = np.random.default_rng(42)
    X = _make_pulse_matrix(num_pulses=16, num_range_bins=64, rng=rng)
    out = clutter(X, rng)
    assert out.shape == X.shape
    assert out.dtype == complex


def test_clutter_adds_power():
    from spectra.impairments.clutter import RadarClutter
    clutter = RadarClutter(cnr=30.0, doppler_spread=100.0, sample_rate=1e6)
    rng = np.random.default_rng(42)
    X = _make_pulse_matrix()
    out = clutter(X, rng)
    assert np.mean(np.abs(out) ** 2) > 0


def test_clutter_deterministic():
    from spectra.impairments.clutter import RadarClutter
    clutter = RadarClutter(cnr=20.0, doppler_spread=50.0, sample_rate=1e6)
    X = _make_pulse_matrix()
    out1 = clutter(X, np.random.default_rng(99))
    out2 = clutter(X, np.random.default_rng(99))
    assert np.allclose(out1, out2)


def test_clutter_range_extent():
    from spectra.impairments.clutter import RadarClutter
    clutter = RadarClutter(
        cnr=30.0, doppler_spread=50.0, sample_rate=1e6, range_extent=(10, 20)
    )
    rng = np.random.default_rng(42)
    X = _make_pulse_matrix(num_pulses=16, num_range_bins=64)
    out = clutter(X, rng)
    assert np.allclose(out[:, :10], 0.0)
    assert np.allclose(out[:, 20:], 0.0)
    assert np.mean(np.abs(out[:, 10:20]) ** 2) > 0


def test_clutter_doppler_spectrum_shape():
    """Clutter should have energy concentrated near doppler_center."""
    from spectra.impairments.clutter import RadarClutter
    clutter = RadarClutter(
        cnr=40.0, doppler_spread=10.0, doppler_center=0.0, sample_rate=1e6
    )
    rng = np.random.default_rng(42)
    X = _make_pulse_matrix(num_pulses=256, num_range_bins=1)
    out = clutter(X, rng)
    spec = np.abs(np.fft.fft(out[:, 0])) ** 2
    dc_bin = 0
    dc_power = spec[dc_bin]
    edge_power = np.mean(spec[len(spec) // 4 : 3 * len(spec) // 4])
    assert dc_power > edge_power * 5, "Clutter spectrum should peak near DC"


def test_ground_preset():
    from spectra.impairments.clutter import RadarClutter
    clutter = RadarClutter.ground(sample_rate=1e6, terrain="rural")
    assert clutter.cnr > 0
    assert clutter.doppler_spread > 0


def test_sea_preset():
    from spectra.impairments.clutter import RadarClutter
    clutter = RadarClutter.sea(sample_rate=1e6, sea_state=3)
    assert clutter.cnr > 0


def test_weather_preset():
    from spectra.impairments.clutter import RadarClutter
    clutter = RadarClutter.weather(sample_rate=1e6, rain_rate_mmhr=10)
    assert clutter.doppler_center != 0.0
