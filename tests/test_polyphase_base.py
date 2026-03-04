"""Tests for _PolyphaseCodeBase shared behavior."""
import numpy as np
import pytest


class TestPolyphaseBase:
    def test_all_polyphase_same_interface(self):
        from spectra.waveforms.polyphase import (
            FrankCode, P1Code, P2Code, P3Code, P4Code,
        )

        for Cls in [FrankCode, P1Code, P2Code, P3Code, P4Code]:
            w = Cls()
            iq = w.generate(512, 1e6, seed=42)
            assert len(iq) > 0
            assert iq.dtype == np.complex64

    def test_output_deterministic(self):
        from spectra.waveforms.polyphase import FrankCode

        w = FrankCode()
        a = w.generate(256, 1e6, seed=7)
        b = w.generate(256, 1e6, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_bandwidth_consistent(self):
        from spectra.waveforms.polyphase import FrankCode, P3Code

        w1 = FrankCode(code_order=4, samples_per_chip=8)
        assert w1.bandwidth(1e6) == 1e6 / 8

        w2 = P3Code(code_length=16, samples_per_chip=4)
        assert w2.bandwidth(1e6) == 1e6 / 4

    def test_p2_requires_even_order(self):
        from spectra.waveforms.polyphase import P2Code

        with pytest.raises(ValueError):
            P2Code(code_order=3)
