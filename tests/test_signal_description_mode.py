"""Tests for the optional `mode` field on SignalDescription."""
from spectra.scene.signal_desc import SignalDescription


def test_mode_defaults_to_none():
    desc = SignalDescription(
        t_start=0.0, t_stop=1e-3, f_low=-5e3, f_high=5e3,
        label="QPSK", snr=10.0,
    )
    assert desc.mode is None


def test_mode_can_be_set():
    desc = SignalDescription(
        t_start=0.0, t_stop=1e-3, f_low=-5e3, f_high=5e3,
        label="PulsedRadar", snr=10.0, mode="track",
    )
    assert desc.mode == "track"


def test_mode_is_optional_positional_not_required():
    # Existing callers that construct SignalDescription positionally with the
    # first six required fields must still work. This is the back-compat check.
    desc = SignalDescription(0.0, 1e-3, -5e3, 5e3, "QPSK", 10.0)
    assert desc.mode is None
    assert desc.label == "QPSK"
