"""Tests for CoherentReceiver and Decoder stubs."""
import numpy as np
import pytest


def test_passthrough_decoder():
    from spectra.receivers.base import PassthroughDecoder
    dec = PassthroughDecoder()
    bits = np.array([0, 1, 1, 0, 1])
    out = dec.decode(bits)
    assert np.array_equal(out, bits)


def test_viterbi_stub_raises():
    from spectra.receivers.base import ViterbiDecoder
    dec = ViterbiDecoder(constraint_length=7, generators=[0o171, 0o133])
    with pytest.raises(NotImplementedError):
        dec.decode(np.array([0, 1, 0]))


def test_ldpc_stub_raises():
    from spectra.receivers.base import LDPCDecoder
    dec = LDPCDecoder(parity_check_matrix=np.eye(4), max_iterations=50)
    with pytest.raises(NotImplementedError):
        dec.decode(np.array([0, 1, 0]))


def test_coherent_receiver_bpsk_noiseless():
    """Noiseless BPSK should demodulate perfectly."""
    from spectra.receivers.coherent import CoherentReceiver
    from spectra.waveforms import BPSK
    from spectra._rust import generate_bpsk_symbols_with_indices, apply_rrc_filter_with_taps
    from spectra.utils.rrc_cache import cached_rrc_taps

    wf = BPSK(samples_per_symbol=8)
    num_sym = 200
    seed = 42

    symbols, tx_indices = generate_bpsk_symbols_with_indices(num_sym, seed)
    taps = cached_rrc_taps(wf.rolloff, wf.filter_span, wf.samples_per_symbol)
    tx_iq = np.array(apply_rrc_filter_with_taps(symbols, taps, wf.samples_per_symbol))

    rx = CoherentReceiver(wf)
    rx_indices, rx_bits = rx.demodulate(tx_iq)

    trim = wf.filter_span
    tx_arr = np.array(tx_indices)[trim:-trim]
    rx_arr = rx_indices[trim : trim + len(tx_arr)]
    assert np.array_equal(rx_arr, tx_arr)


def test_coherent_receiver_qpsk_noiseless():
    from spectra.receivers.coherent import CoherentReceiver
    from spectra.waveforms import QPSK
    from spectra._rust import generate_qpsk_symbols_with_indices, apply_rrc_filter_with_taps
    from spectra.utils.rrc_cache import cached_rrc_taps

    wf = QPSK(samples_per_symbol=8)
    num_sym = 200
    seed = 99

    symbols, tx_indices = generate_qpsk_symbols_with_indices(num_sym, seed)
    taps = cached_rrc_taps(wf.rolloff, wf.filter_span, wf.samples_per_symbol)
    tx_iq = np.array(apply_rrc_filter_with_taps(symbols, taps, wf.samples_per_symbol))

    rx = CoherentReceiver(wf)
    rx_indices, rx_bits = rx.demodulate(tx_iq)

    trim = wf.filter_span
    tx_arr = np.array(tx_indices)[trim:-trim]
    rx_arr = rx_indices[trim : trim + len(tx_arr)]
    assert np.array_equal(rx_arr, tx_arr)


def test_coherent_receiver_output_shapes():
    from spectra.receivers.coherent import CoherentReceiver
    from spectra.waveforms import QPSK
    from spectra._rust import generate_qpsk_symbols_with_indices, apply_rrc_filter_with_taps
    from spectra.utils.rrc_cache import cached_rrc_taps

    wf = QPSK(samples_per_symbol=8)
    symbols, tx_indices = generate_qpsk_symbols_with_indices(100, 42)
    taps = cached_rrc_taps(wf.rolloff, wf.filter_span, wf.samples_per_symbol)
    tx_iq = np.array(apply_rrc_filter_with_taps(symbols, taps, wf.samples_per_symbol))

    rx = CoherentReceiver(wf)
    rx_indices, rx_bits = rx.demodulate(tx_iq)

    assert rx_indices.ndim == 1
    assert rx_bits.ndim == 1
    assert len(rx_bits) == len(rx_indices) * 2


def test_constellation_to_bits():
    from spectra.receivers.coherent import constellation_to_bits
    bits = constellation_to_bits(np.array([0, 1, 0, 1], dtype=np.uint32), 2)
    assert len(bits) == 4
    assert bits[0] == 0
    assert bits[1] == 1


def test_receiver_abc():
    from spectra.receivers.base import Receiver
    with pytest.raises(TypeError):
        Receiver()
