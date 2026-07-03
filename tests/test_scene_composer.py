import numpy as np
import pytest


class TestSceneConfig:
    def test_creation(self):
        from spectra.scene.composer import SceneConfig
        from spectra.waveforms import BPSK, QPSK

        config = SceneConfig(
            capture_duration=1e-3,
            capture_bandwidth=20e6,
            sample_rate=40e6,
            num_signals=(2, 5),
            signal_pool=[QPSK(), BPSK()],
            snr_range=(5, 25),
            allow_overlap=True,
        )
        assert config.capture_duration == 1e-3
        assert config.sample_rate == 40e6


class TestComposer:
    @pytest.fixture
    def basic_config(self):
        from spectra.scene.composer import SceneConfig
        from spectra.waveforms import BPSK, QPSK

        return SceneConfig(
            capture_duration=1e-3,
            capture_bandwidth=1e6,
            sample_rate=2e6,
            num_signals=(2, 4),
            signal_pool=[QPSK(samples_per_symbol=4), BPSK(samples_per_symbol=4)],
            snr_range=(10, 20),
            allow_overlap=True,
        )

    def test_generate_returns_iq_and_descs(self, basic_config):
        from spectra.scene.composer import Composer

        composer = Composer(basic_config)
        iq, descs = composer.generate(seed=42)
        assert isinstance(iq, np.ndarray)
        assert iq.dtype == np.complex64
        assert isinstance(descs, list)
        assert len(descs) >= 2

    def test_iq_length_matches_config(self, basic_config):
        from spectra.scene.composer import Composer

        composer = Composer(basic_config)
        iq, _ = composer.generate(seed=42)
        expected_len = int(basic_config.capture_duration * basic_config.sample_rate)
        assert len(iq) == expected_len

    def test_signal_descs_have_required_fields(self, basic_config):
        from spectra.scene.composer import Composer

        composer = Composer(basic_config)
        _, descs = composer.generate(seed=42)
        for desc in descs:
            assert hasattr(desc, "t_start")
            assert hasattr(desc, "t_stop")
            assert hasattr(desc, "f_low")
            assert hasattr(desc, "f_high")
            assert hasattr(desc, "label")
            assert hasattr(desc, "snr")

    def test_signals_within_capture_bounds(self, basic_config):
        from spectra.scene.composer import Composer

        composer = Composer(basic_config)
        _, descs = composer.generate(seed=42)
        half_bw = basic_config.capture_bandwidth / 2
        for desc in descs:
            assert desc.t_start >= 0.0
            assert desc.t_stop <= basic_config.capture_duration
            assert desc.f_low >= -half_bw
            assert desc.f_high <= half_bw

    def test_deterministic_with_seed(self, basic_config):
        from spectra.scene.composer import Composer

        composer = Composer(basic_config)
        iq1, descs1 = composer.generate(seed=42)
        iq2, descs2 = composer.generate(seed=42)
        np.testing.assert_array_equal(iq1, iq2)
        assert len(descs1) == len(descs2)
        for d1, d2 in zip(descs1, descs2):
            assert d1.label == d2.label
            assert d1.t_start == d2.t_start

    def test_multiple_signals_present(self, basic_config):
        from spectra.scene.composer import Composer

        composer = Composer(basic_config)
        _, descs = composer.generate(seed=42)
        assert len(descs) >= 2

    def test_fixed_num_signals(self):
        from spectra.scene.composer import Composer, SceneConfig
        from spectra.waveforms import QPSK

        config = SceneConfig(
            capture_duration=1e-3,
            capture_bandwidth=1e6,
            sample_rate=2e6,
            num_signals=3,
            signal_pool=[QPSK(samples_per_symbol=4)],
            snr_range=(10, 20),
            allow_overlap=True,
        )
        composer = Composer(config)
        _, descs = composer.generate(seed=42)
        assert len(descs) == 3


def _welch_psd(iq, sample_rate, nperseg=1024):
    win = np.hanning(nperseg)
    step = nperseg // 2
    nseg = (len(iq) - nperseg) // step + 1
    psd = np.zeros(nperseg)
    for k in range(nseg):
        seg = np.asarray(iq[k * step : k * step + nperseg], dtype=np.complex128) * win
        psd += np.abs(np.fft.fft(seg)) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(nperseg, 1.0 / sample_rate))
    return freqs, np.fft.fftshift(psd / nseg)


class TestComposerOffsetWaveforms:
    """Ground-truth boxes must track the measured occupied band even for
    waveforms whose baseband occupancy is not centered at DC (e.g. OFDM
    with asymmetric guard bands)."""

    FS = 10e6
    FFT = 256

    @pytest.fixture
    def offset_config(self):
        from spectra.scene.composer import SceneConfig
        from spectra.waveforms.ofdm import OFDM

        return SceneConfig(
            capture_duration=5e-3,
            capture_bandwidth=8e6,
            sample_rate=self.FS,
            num_signals=1,
            signal_pool=[OFDM(num_subcarriers=64, fft_size=self.FFT, guard_bands=(16, 0))],
            snr_range=(10, 10),
            allow_overlap=True,
        )

    def test_box_matches_measured_occupied_band(self, offset_config):
        from spectra.scene.composer import Composer

        composer = Composer(offset_config)
        iq, descs = composer.generate(seed=42)
        assert len(descs) == 1
        desc = descs[0]

        freqs, psd = _welch_psd(iq, self.FS)
        occupied = freqs[psd > 1e-2 * np.max(psd)]  # -20 dB threshold
        measured_center = (occupied.max() + occupied.min()) / 2.0
        box_center = (desc.f_low + desc.f_high) / 2.0

        subcarrier_spacing = self.FS / self.FFT
        assert measured_center == pytest.approx(box_center, abs=2 * subcarrier_spacing)

    def test_box_width_matches_measured_extent(self, offset_config):
        from spectra.scene.composer import Composer

        composer = Composer(offset_config)
        iq, descs = composer.generate(seed=42)
        desc = descs[0]

        freqs, psd = _welch_psd(iq, self.FS)
        shifted_cum = np.cumsum(psd) / np.sum(psd)
        lo = freqs[np.searchsorted(shifted_cum, 0.005)]
        hi = freqs[np.searchsorted(shifted_cum, 0.995)]
        assert (hi - lo) == pytest.approx(desc.f_high - desc.f_low, rel=0.10)
