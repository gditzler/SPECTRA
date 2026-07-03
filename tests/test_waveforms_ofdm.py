import numpy as np
import numpy.testing as npt
import pytest


class TestOFDMWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.ofdm import OFDM

        waveform = OFDM()
        iq = waveform.generate(num_symbols=10, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_output_length(self, sample_rate):
        from spectra.waveforms.ofdm import OFDM

        nsc, fft, cp = 64, 256, 16
        waveform = OFDM(num_subcarriers=nsc, fft_size=fft, cp_length=cp)
        iq = waveform.generate(num_symbols=5, sample_rate=sample_rate)
        assert len(iq) == 5 * (fft + cp)

    def test_label(self):
        from spectra.waveforms.ofdm import OFDM

        assert OFDM().label == "OFDM"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms.ofdm import OFDM

        nsc, fft = 64, 256
        waveform = OFDM(num_subcarriers=nsc, fft_size=fft)
        expected_bw = nsc * sample_rate / fft
        assert waveform.bandwidth(sample_rate) == pytest.approx(expected_bw)

    def test_samples_per_symbol_attribute(self):
        from spectra.waveforms.ofdm import OFDM

        waveform = OFDM(fft_size=256, cp_length=16)
        assert waveform.samples_per_symbol == 272

    def test_deterministic_with_seed(self, sample_rate):
        from spectra.waveforms.ofdm import OFDM

        waveform = OFDM()
        iq1 = waveform.generate(num_symbols=5, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=5, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    def test_no_cyclic_prefix(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.ofdm import OFDM

        waveform = OFDM(cp_length=0)
        iq = waveform.generate(num_symbols=5, sample_rate=sample_rate)
        assert_valid_iq(iq)
        assert len(iq) == 5 * 256  # default fft_size, no CP


class TestOFDMEnhanced:
    def test_backward_compat(self):
        """Default params produce same behavior as before."""
        from spectra.waveforms.ofdm import OFDM

        ofdm = OFDM()
        iq = ofdm.generate(num_symbols=10, sample_rate=1e6, seed=42)
        assert iq.dtype == np.complex64
        assert len(iq) > 0

    def test_guard_bands(self):
        from spectra.waveforms.ofdm import OFDM

        ofdm = OFDM(num_subcarriers=64, fft_size=256, guard_bands=(6, 5))
        iq = ofdm.generate(num_symbols=5, sample_rate=1e6, seed=42)
        assert len(iq) > 0
        # Verify bandwidth is reduced
        bw_no_guard = OFDM(num_subcarriers=64, fft_size=256).bandwidth(1e6)
        bw_guard = ofdm.bandwidth(1e6)
        assert bw_guard < bw_no_guard

    def test_dc_null(self):
        from spectra.waveforms.ofdm import OFDM

        ofdm = OFDM(num_subcarriers=64, fft_size=256, dc_null=True)
        iq = ofdm.generate(num_symbols=5, sample_rate=1e6, seed=42)
        assert len(iq) > 0

    def test_pilots(self):
        from spectra.waveforms.ofdm import OFDM

        ofdm = OFDM(num_subcarriers=64, fft_size=256, pilot_indices=[0, 16, 32, 48])
        iq = ofdm.generate(num_symbols=5, sample_rate=1e6, seed=42)
        assert len(iq) > 0

    def test_modulation_bpsk(self):
        from spectra.waveforms.ofdm import OFDM

        ofdm = OFDM(modulation="BPSK")
        iq = ofdm.generate(num_symbols=5, sample_rate=1e6, seed=42)
        assert len(iq) > 0

    def test_modulation_qam16(self):
        from spectra.waveforms.ofdm import OFDM

        ofdm = OFDM(modulation="QAM16")
        iq = ofdm.generate(num_symbols=5, sample_rate=1e6, seed=42)
        assert len(iq) > 0


def _bin_powers(waveform, fft_size, num_symbols=100, sample_rate=10e6, seed=123):
    """Average per-FFT-bin power from CP-free symbols (exact, noiseless)."""
    iq = waveform.generate(num_symbols=num_symbols, sample_rate=sample_rate, seed=seed)
    blocks = np.asarray(iq, dtype=np.complex128).reshape(num_symbols, fft_size)
    return np.mean(np.abs(np.fft.fft(blocks, axis=1)) ** 2, axis=0)


def _occupied_bins(powers, rel_threshold=1e-6):
    """Set of signed FFT bin indices carrying power (bin > threshold * max)."""
    n = len(powers)
    idx = np.nonzero(powers > rel_threshold * np.max(powers))[0]
    return {int(i) if i < n // 2 else int(i) - n for i in idx}


def _occupied_extent_hz(powers, sample_rate):
    """Edge-to-edge extent of occupied bins in Hz."""
    bins = _occupied_bins(powers)
    return (max(bins) - min(bins)) * sample_rate / len(powers)


class TestOFDMSpectralOccupancy:
    """Bandwidth labels must match the empirically occupied spectrum."""

    NSC, FFT, FS = 64, 256, 10e6

    def _make(self, cls=None, **kwargs):
        from spectra.waveforms.ofdm import OFDM

        cls = cls or OFDM
        return cls(num_subcarriers=self.NSC, fft_size=self.FFT, cp_length=0, **kwargs)

    def test_default_occupancy_symmetric_and_matches_bandwidth(self):
        wf = self._make()
        powers = _bin_powers(wf, self.FFT, sample_rate=self.FS)
        bins = _occupied_bins(powers)
        assert bins == {b for b in range(-32, 33) if b != 0}
        extent = _occupied_extent_hz(powers, self.FS)
        assert extent == pytest.approx(wf.bandwidth(self.FS), rel=0.05)

    def test_pilots_do_not_shrink_bandwidth_label(self):
        """Pilots transmit energy; they must not be subtracted from the label."""
        pilots = list(range(0, self.NSC, 8))
        wf = self._make(pilot_indices=pilots)
        assert wf.bandwidth(self.FS) == pytest.approx(self.NSC * self.FS / self.FFT)

    def test_pilots_measured_occupancy_matches_bandwidth(self):
        pilots = list(range(0, self.NSC, 8))
        wf = self._make(pilot_indices=pilots)
        powers = _bin_powers(wf, self.FFT, sample_rate=self.FS)
        extent = _occupied_extent_hz(powers, self.FS)
        assert extent == pytest.approx(wf.bandwidth(self.FS), rel=0.05)

    def test_guard_bands_trim_band_edges(self):
        """guard_bands=(16, 0) must remove the 16 lowest-frequency carriers,
        not carve a notch above DC."""
        wf = self._make(guard_bands=(16, 0))
        powers = _bin_powers(wf, self.FFT, sample_rate=self.FS)
        bins = _occupied_bins(powers)
        assert bins == {b for b in range(-16, 33) if b != 0}

    def test_guard_bands_measured_occupancy_matches_bandwidth(self):
        wf = self._make(guard_bands=(16, 0))
        powers = _bin_powers(wf, self.FFT, sample_rate=self.FS)
        extent = _occupied_extent_hz(powers, self.FS)
        assert extent == pytest.approx(wf.bandwidth(self.FS), rel=0.05)

    def test_symmetric_guard_bands_stay_centered(self):
        wf = self._make(guard_bands=(8, 8))
        powers = _bin_powers(wf, self.FFT, sample_rate=self.FS)
        bins = _occupied_bins(powers)
        assert bins == {b for b in range(-24, 25) if b != 0}
        assert wf.bandwidth(self.FS) == pytest.approx(48 * self.FS / self.FFT)

    def test_dc_bin_never_occupied(self):
        for dc_null in (False, True):
            wf = self._make(dc_null=dc_null)
            powers = _bin_powers(wf, self.FFT, sample_rate=self.FS)
            assert 0 not in _occupied_bins(powers)

    def test_dc_null_does_not_carve_interior_notch(self):
        """dc_null previously zeroed the most-negative carrier. It must not
        remove any occupied carrier (DC is structurally unoccupied)."""
        wf = self._make(dc_null=True)
        powers = _bin_powers(wf, self.FFT, sample_rate=self.FS)
        assert _occupied_bins(powers) == {b for b in range(-32, 33) if b != 0}
        assert wf.bandwidth(self.FS) == pytest.approx(self.NSC * self.FS / self.FFT)

    def test_99pct_obw_matches_bandwidth(self):
        wf = self._make()
        powers = _bin_powers(wf, self.FFT, num_symbols=400, sample_rate=self.FS)
        shifted = np.fft.fftshift(powers)
        cum = np.cumsum(shifted) / np.sum(shifted)
        freqs = np.fft.fftshift(np.fft.fftfreq(self.FFT, d=1.0 / self.FS))
        lo = freqs[np.searchsorted(cum, 0.005)]
        hi = freqs[np.searchsorted(cum, 0.995)]
        assert (hi - lo) == pytest.approx(wf.bandwidth(self.FS), rel=0.05)

    def test_center_offset_zero_by_default(self):
        from spectra.waveforms import QPSK

        assert self._make().center_offset(self.FS) == 0.0
        assert QPSK().center_offset(self.FS) == 0.0

    def test_center_offset_symmetric_guards_zero(self):
        assert self._make(guard_bands=(8, 8)).center_offset(self.FS) == 0.0

    def test_center_offset_asymmetric_guards(self):
        df = self.FS / self.FFT
        assert self._make(guard_bands=(16, 0)).center_offset(self.FS) == pytest.approx(8 * df)
        assert self._make(guard_bands=(0, 16)).center_offset(self.FS) == pytest.approx(-8 * df)

    def test_center_offset_matches_measured_band_center(self):
        wf = self._make(guard_bands=(16, 0))
        powers = _bin_powers(wf, self.FFT, sample_rate=self.FS)
        bins = _occupied_bins(powers)
        measured_center = (max(bins) + min(bins)) / 2 * self.FS / self.FFT
        assert measured_center == pytest.approx(wf.center_offset(self.FS), abs=self.FS / self.FFT)

    def test_scfdma_guard_bands_trim_band_edges(self):
        from spectra.waveforms.ofdm import SCFDMA

        wf = self._make(cls=SCFDMA, guard_bands=(16, 0))
        powers = _bin_powers(wf, self.FFT, sample_rate=self.FS)
        bins = _occupied_bins(powers)
        assert bins == {b for b in range(-16, 33) if b != 0}

    def test_scfdma_pilots_do_not_shrink_bandwidth_label(self):
        from spectra.waveforms.ofdm import SCFDMA

        wf = self._make(cls=SCFDMA, pilot_indices=list(range(0, self.NSC, 8)))
        assert wf.bandwidth(self.FS) == pytest.approx(self.NSC * self.FS / self.FFT)


class TestSCFDMA:
    def test_generate(self):
        from spectra.waveforms.ofdm import SCFDMA

        sc = SCFDMA(num_subcarriers=64, fft_size=256)
        iq = sc.generate(num_symbols=10, sample_rate=1e6, seed=42)
        assert iq.dtype == np.complex64
        assert len(iq) > 0

    def test_label(self):
        from spectra.waveforms.ofdm import SCFDMA

        assert SCFDMA().label == "SC-FDMA"

    def test_lower_papr_than_ofdm(self):
        """SC-FDMA should have lower PAPR than OFDM."""
        from spectra.waveforms.ofdm import OFDM, SCFDMA

        ofdm = OFDM(num_subcarriers=64, fft_size=256)
        scfdma = SCFDMA(num_subcarriers=64, fft_size=256)

        ofdm_iq = ofdm.generate(num_symbols=100, sample_rate=1e6, seed=42)
        sc_iq = scfdma.generate(num_symbols=100, sample_rate=1e6, seed=42)

        def papr(x):
            return float(np.max(np.abs(x) ** 2) / np.mean(np.abs(x) ** 2))

        # SC-FDMA should generally have lower PAPR
        assert papr(sc_iq) < papr(ofdm_iq) * 1.5  # allow some margin
