import numpy as np
import pytest

from spectra.scene.signal_desc import SignalDescription


def make_signal_description(
    t_start: float = 0.0,
    t_stop: float = 0.001,
    f_low: float = -5e3,
    f_high: float = 5e3,
    label: str = "QPSK",
    snr: float = 20.0,
    **kwargs,
) -> SignalDescription:
    """Create a SignalDescription with test defaults."""
    return SignalDescription(
        t_start=t_start,
        t_stop=t_stop,
        f_low=f_low,
        f_high=f_high,
        label=label,
        snr=snr,
        **kwargs,
    )


@pytest.fixture
def signal_description():
    """Default SignalDescription for impairment tests."""
    return make_signal_description()


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def sample_rate():
    """Default sample rate for tests."""
    return 1e6


@pytest.fixture
def assert_valid_iq():
    """Assert that an IQ array is well-formed."""

    def _check(iq, expected_length=None):
        assert isinstance(iq, np.ndarray), f"Expected ndarray, got {type(iq)}"
        assert iq.dtype == np.complex64, f"Expected complex64, got {iq.dtype}"
        assert iq.ndim == 1, f"Expected 1D array, got {iq.ndim}D"
        assert not np.any(np.isnan(iq)), "Array contains NaN values"
        assert not np.any(np.isinf(iq)), "Array contains Inf values"
        if expected_length is not None:
            assert len(iq) == expected_length, f"Expected length {expected_length}, got {len(iq)}"

    return _check
