import numpy as np
import numpy.testing as npt
import pytest

from spectra.scene.signal_desc import SignalDescription


def _make_desc():
    return SignalDescription(0.0, 0.001, -5e3, 5e3, "QPSK", 20.0)


class TestPhaseOffset:
    def test_fixed_offset_rotates_signal(self):
        from spectra.impairments.phase_offset import PhaseOffset

        iq = np.ones(1024, dtype=np.complex64)
        desc = _make_desc()
        rotated, _ = PhaseOffset(offset=np.pi / 4)(iq, desc)
        expected = np.exp(1j * np.pi / 4).astype(np.complex64)
        npt.assert_allclose(rotated, expected, atol=1e-6)

    def test_zero_offset_preserves_signal(self):
        from spectra.impairments.phase_offset import PhaseOffset

        iq = np.ones(512, dtype=np.complex64) * (1 + 1j) / np.sqrt(2)
        desc = _make_desc()
        result, _ = PhaseOffset(offset=0.0)(iq, desc)
        npt.assert_allclose(result, iq, atol=1e-6)

    def test_magnitude_preserved(self):
        from spectra.impairments.phase_offset import PhaseOffset

        rng = np.random.default_rng(42)
        iq = (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(
            np.complex64
        )
        desc = _make_desc()
        rotated, _ = PhaseOffset(offset=1.23)(iq, desc)
        npt.assert_allclose(np.abs(rotated), np.abs(iq), atol=1e-5)

    def test_max_offset_randomizes(self):
        from spectra.impairments.phase_offset import PhaseOffset

        iq = np.ones(512, dtype=np.complex64)
        desc = _make_desc()
        po = PhaseOffset(max_offset=np.pi)
        results = [po(iq.copy(), desc)[0][0] for _ in range(20)]
        phases = [np.angle(r) for r in results]
        assert max(phases) - min(phases) > 0.5, "Random phases should vary"

    def test_output_shape_and_dtype(self):
        from spectra.impairments.phase_offset import PhaseOffset

        iq = np.ones(256, dtype=np.complex64)
        desc = _make_desc()
        result, _ = PhaseOffset(offset=0.5)(iq, desc)
        assert result.shape == iq.shape
        assert result.dtype == np.complex64

    def test_desc_unchanged(self):
        from spectra.impairments.phase_offset import PhaseOffset

        iq = np.ones(256, dtype=np.complex64)
        desc = _make_desc()
        _, new_desc = PhaseOffset(offset=0.5)(iq, desc)
        assert new_desc.f_low == desc.f_low
        assert new_desc.f_high == desc.f_high

    def test_requires_offset_or_max_offset(self):
        from spectra.impairments.phase_offset import PhaseOffset

        with pytest.raises(ValueError):
            PhaseOffset()
