import numpy as np
import numpy.testing as npt
import pytest


class TestOOKWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import OOK

        iq = OOK().generate(num_symbols=128, sample_rate=sample_rate, seed=42)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms import OOK

        assert OOK().label == "OOK"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms import OOK

        sps = 8
        rolloff = 0.35
        wf = OOK(samples_per_symbol=sps, rolloff=rolloff)
        expected = (sample_rate / sps) * (1.0 + rolloff)
        assert wf.bandwidth(sample_rate) == pytest.approx(expected)

    def test_deterministic(self, sample_rate):
        from spectra.waveforms import OOK

        wf = OOK()
        iq1 = wf.generate(64, sample_rate, seed=42)
        iq2 = wf.generate(64, sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)


@pytest.mark.parametrize(
    "cls_name,order,label",
    [
        ("ASK4", 4, "4ASK"),
        ("ASK8", 8, "8ASK"),
        ("ASK16", 16, "16ASK"),
        ("ASK32", 32, "32ASK"),
        ("ASK64", 64, "64ASK"),
    ],
)
class TestASKVariants:
    def test_generate_valid_iq(self, cls_name, order, label, assert_valid_iq, sample_rate):
        import spectra.waveforms as wf_mod

        cls = getattr(wf_mod, cls_name)
        iq = cls().generate(128, sample_rate, seed=42)
        assert_valid_iq(iq)

    def test_label(self, cls_name, order, label):
        import spectra.waveforms as wf_mod

        cls = getattr(wf_mod, cls_name)
        assert cls().label == label


@pytest.mark.rust
class TestASKSymbols:
    def test_ook_symbol_values(self):
        from spectra._rust import generate_ask_symbols

        symbols = generate_ask_symbols(1000, 2, seed=0)
        unique = np.unique(symbols.real)
        # OOK: two unique real levels (0 and scaled 1)
        assert len(unique) == 2
        # Imaginary should be zero
        npt.assert_allclose(symbols.imag, 0.0, atol=1e-6)

    def test_4ask_four_levels(self):
        from spectra._rust import generate_ask_symbols

        symbols = generate_ask_symbols(10000, 4, seed=0)
        unique = np.unique(np.round(symbols.real, 4))
        assert len(unique) == 4
