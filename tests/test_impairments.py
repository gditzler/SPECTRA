import numpy as np
import numpy.testing as npt
import pytest


class TestSignalDescription:
    def test_creation(self):
        from spectra.scene.signal_desc import SignalDescription
        desc = SignalDescription(
            t_start=0.0, t_stop=0.001,
            f_low=-5e3, f_high=5e3,
            label="QPSK", snr=20.0,
        )
        assert desc.t_start == 0.0
        assert desc.label == "QPSK"

    def test_f_center(self):
        from spectra.scene.signal_desc import SignalDescription
        desc = SignalDescription(
            t_start=0.0, t_stop=0.001,
            f_low=-5e3, f_high=5e3,
            label="QPSK", snr=20.0,
        )
        assert desc.f_center == pytest.approx(0.0)

    def test_bandwidth_property(self):
        from spectra.scene.signal_desc import SignalDescription
        desc = SignalDescription(
            t_start=0.0, t_stop=0.001,
            f_low=-5e3, f_high=5e3,
            label="QPSK", snr=20.0,
        )
        assert desc.bandwidth == pytest.approx(10e3)


class TestAWGN:
    def test_adds_noise(self, sample_rate):
        from spectra.impairments import AWGN
        from spectra.scene.signal_desc import SignalDescription
        iq = np.ones(1024, dtype=np.complex64)
        desc = SignalDescription(0.0, 0.001, -5e3, 5e3, "QPSK", 20.0)
        noisy_iq, _ = AWGN(snr=10.0)(iq, desc)
        assert not np.array_equal(iq, noisy_iq)

    def test_output_shape_preserved(self):
        from spectra.impairments import AWGN
        from spectra.scene.signal_desc import SignalDescription
        iq = np.ones(512, dtype=np.complex64)
        desc = SignalDescription(0.0, 0.001, -5e3, 5e3, "QPSK", 20.0)
        noisy_iq, _ = AWGN(snr=20.0)(iq, desc)
        assert noisy_iq.shape == iq.shape
        assert noisy_iq.dtype == np.complex64

    def test_snr_range_randomizes(self):
        from spectra.impairments import AWGN
        from spectra.scene.signal_desc import SignalDescription
        iq = np.ones(1024, dtype=np.complex64)
        desc = SignalDescription(0.0, 0.001, -5e3, 5e3, "QPSK", 20.0)
        awgn = AWGN(snr_range=(0, 30))
        results = [awgn(iq.copy(), desc)[0] for _ in range(10)]
        # Different noise levels should produce different outputs
        diffs = [np.sum(np.abs(results[i] - results[i+1])) for i in range(9)]
        assert not all(d == 0 for d in diffs)

    def test_high_snr_preserves_signal(self):
        from spectra.impairments import AWGN
        from spectra.scene.signal_desc import SignalDescription
        iq = np.ones(4096, dtype=np.complex64)
        desc = SignalDescription(0.0, 0.001, -5e3, 5e3, "QPSK", 20.0)
        noisy_iq, _ = AWGN(snr=60.0)(iq, desc)
        npt.assert_allclose(iq, noisy_iq, atol=0.01)


class TestFrequencyOffset:
    def test_applies_offset(self, sample_rate):
        from spectra.impairments import FrequencyOffset
        from spectra.scene.signal_desc import SignalDescription
        iq = np.ones(1024, dtype=np.complex64)
        desc = SignalDescription(0.0, 1024 / sample_rate, -5e3, 5e3, "QPSK", 20.0)
        offset_iq, new_desc = FrequencyOffset(offset=100.0)(iq, desc, sample_rate=sample_rate)
        # Magnitude should be preserved
        npt.assert_allclose(np.abs(offset_iq), 1.0, atol=1e-5)
        # Frequency content should shift
        assert not np.allclose(iq, offset_iq)

    def test_updates_signal_desc(self, sample_rate):
        from spectra.impairments import FrequencyOffset
        from spectra.scene.signal_desc import SignalDescription
        iq = np.ones(1024, dtype=np.complex64)
        desc = SignalDescription(0.0, 0.001, -5e3, 5e3, "QPSK", 20.0)
        _, new_desc = FrequencyOffset(offset=100.0)(iq, desc, sample_rate=sample_rate)
        assert new_desc.f_low == pytest.approx(-5e3 + 100.0)
        assert new_desc.f_high == pytest.approx(5e3 + 100.0)


class TestCompose:
    def test_chains_transforms(self, sample_rate):
        from spectra.impairments import AWGN, FrequencyOffset, Compose
        from spectra.scene.signal_desc import SignalDescription
        iq = np.ones(1024, dtype=np.complex64)
        desc = SignalDescription(0.0, 1024 / sample_rate, -5e3, 5e3, "QPSK", 20.0)
        chain = Compose([
            FrequencyOffset(offset=100.0),
            AWGN(snr=20.0),
        ])
        result_iq, result_desc = chain(iq, desc, sample_rate=sample_rate)
        assert result_iq.shape == iq.shape
        assert result_desc.f_low == pytest.approx(-5e3 + 100.0)

    def test_empty_compose(self):
        from spectra.impairments import Compose
        from spectra.scene.signal_desc import SignalDescription
        iq = np.ones(512, dtype=np.complex64)
        desc = SignalDescription(0.0, 0.001, -5e3, 5e3, "QPSK", 20.0)
        chain = Compose([])
        result_iq, result_desc = chain(iq, desc)
        npt.assert_array_equal(iq, result_iq)
