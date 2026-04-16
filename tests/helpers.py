"""Shared test utilities for SPECTRA tests."""

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
