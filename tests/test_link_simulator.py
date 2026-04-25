"""Tests for LinkSimulator."""
import numpy as np


def test_link_results_fields():
    from spectra.link.results import LinkResults
    r = LinkResults(
        eb_n0_db=np.array([0, 5, 10]),
        ber=np.array([0.1, 0.01, 0.001]),
        ser=np.array([0.2, 0.02, 0.002]),
        per=np.array([0.5, 0.1, 0.01]),
        num_bits=10000,
        num_symbols=5000,
        packet_length=1000,
        waveform_label="BPSK",
    )
    assert len(r.eb_n0_db) == 3
    assert r.waveform_label == "BPSK"


def test_theoretical_ber_bpsk():
    from spectra.link.results import LinkResults
    r = LinkResults(
        eb_n0_db=np.array([0, 5, 10]),
        ber=np.zeros(3), ser=np.zeros(3), per=np.zeros(3),
        num_bits=1000, num_symbols=1000, packet_length=100,
        waveform_label="BPSK",
    )
    theory = r.theoretical_ber()
    assert theory is not None
    assert len(theory) == 3
    assert theory[2] < 1e-4


def test_theoretical_ber_qpsk_returns_none():
    from spectra.link.results import LinkResults
    r = LinkResults(
        eb_n0_db=np.array([0, 5, 10]),
        ber=np.zeros(3), ser=np.zeros(3), per=np.zeros(3),
        num_bits=1000, num_symbols=500, packet_length=100,
        waveform_label="QPSK",
    )
    assert r.theoretical_ber() is None


def test_link_simulator_bpsk_low_snr():
    from spectra.link.simulator import LinkSimulator
    from spectra.waveforms import BPSK
    sim = LinkSimulator(waveform=BPSK(samples_per_symbol=8), num_symbols=2000, seed=42)
    results = sim.run(np.array([0.0]))
    assert results.ber[0] > 0.01


def test_link_simulator_bpsk_high_snr():
    from spectra.link.simulator import LinkSimulator
    from spectra.waveforms import BPSK
    sim = LinkSimulator(waveform=BPSK(samples_per_symbol=8), num_symbols=2000, seed=42)
    results = sim.run(np.array([15.0]))
    assert results.ber[0] < 0.01


def test_link_simulator_sweep():
    from spectra.link.simulator import LinkSimulator
    from spectra.waveforms import BPSK
    sim = LinkSimulator(waveform=BPSK(samples_per_symbol=8), num_symbols=1000, seed=42)
    eb_n0 = np.array([0.0, 5.0, 10.0])
    results = sim.run(eb_n0)
    assert len(results.ber) == 3
    assert len(results.ser) == 3
    assert len(results.per) == 3
    assert results.ber[0] >= results.ber[1]
    assert results.ber[1] >= results.ber[2]


def test_link_simulator_deterministic():
    from spectra.link.simulator import LinkSimulator
    from spectra.waveforms import BPSK
    sim1 = LinkSimulator(waveform=BPSK(samples_per_symbol=8), num_symbols=500, seed=42)
    sim2 = LinkSimulator(waveform=BPSK(samples_per_symbol=8), num_symbols=500, seed=42)
    r1 = sim1.run(np.array([5.0]))
    r2 = sim2.run(np.array([5.0]))
    assert np.allclose(r1.ber, r2.ber)


def test_link_simulator_qam16():
    from spectra.link.simulator import LinkSimulator
    from spectra.waveforms import QAM16
    sim = LinkSimulator(waveform=QAM16(samples_per_symbol=8), num_symbols=1000, seed=42)
    results = sim.run(np.array([15.0]))
    assert isinstance(results.ber[0], float)
    assert results.waveform_label == "16QAM"
