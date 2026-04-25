# tests/test_studio_params.py
"""Tests for waveform parameter registry."""


def test_get_waveform_params_bpsk():
    from spectra.studio.params import get_waveform_params
    params = get_waveform_params("BPSK")
    assert isinstance(params, list)
    # RRC waveforms should have samples_per_symbol
    names = [p["name"] for p in params]
    assert "samples_per_symbol" in names


def test_get_waveform_params_lfm():
    from spectra.studio.params import get_waveform_params
    params = get_waveform_params("LFM")
    assert isinstance(params, list)
    assert len(params) > 0


def test_get_waveform_params_pulsed_radar():
    from spectra.studio.params import get_waveform_params
    params = get_waveform_params("PulsedRadar")
    names = [p["name"] for p in params]
    assert "pri_samples" in names
    assert "num_pulses" in names
    assert "pulse_shape" in names


def test_get_waveform_params_unknown():
    from spectra.studio.params import get_waveform_params
    params = get_waveform_params("NONEXISTENT")
    assert params == []


def test_get_all_categories():
    from spectra.studio.params import get_all_categories
    cats = get_all_categories()
    assert "PSK" in cats
    assert "Radar" in cats
    assert len(cats) >= 10


def test_get_waveforms_for_category():
    from spectra.studio.params import get_waveforms_for_category
    psk = get_waveforms_for_category("PSK")
    assert "BPSK" in psk
    assert "QPSK" in psk
