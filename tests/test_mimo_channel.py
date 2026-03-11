"""Tests for MIMO channel impairment."""

import numpy as np
import pytest
from spectra.impairments.mimo_channel import MIMOChannel
from spectra.scene.signal_desc import SignalDescription


def _make_desc():
    return SignalDescription(
        t_start=0.0,
        t_stop=1.0,
        f_low=-500e3,
        f_high=500e3,
        label="test",
        snr=20.0,
    )


class TestMIMOFlat:
    """Flat fading MIMO channel tests."""

    def test_2x4_shape(self):
        """2 TX, 4 RX: input (2, 1024) -> output (4, 1024)."""
        ch = MIMOChannel(n_tx=2, n_rx=4, channel_type="flat")
        iq = (np.random.randn(2, 1024) + 1j * np.random.randn(2, 1024)).astype(np.complex64)
        out, desc = ch(iq, _make_desc(), sample_rate=1e6)
        assert out.shape == (4, 1024)
        assert out.dtype == np.complex64

    def test_1d_auto_expansion(self):
        """1D input should be auto-expanded for single-TX."""
        ch = MIMOChannel(n_tx=1, n_rx=4, channel_type="flat")
        iq = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
        out, desc = ch(iq, _make_desc(), sample_rate=1e6)
        assert out.shape == (4, 1024)

    def test_1d_multi_tx_replication(self):
        """1D input with n_tx > 1 should replicate to all TX."""
        ch = MIMOChannel(n_tx=2, n_rx=2, channel_type="flat")
        iq = (np.random.randn(512) + 1j * np.random.randn(512)).astype(np.complex64)
        out, desc = ch(iq, _make_desc(), sample_rate=1e6)
        assert out.shape == (2, 512)

    def test_wrong_tx_count_raises(self):
        """Input with wrong TX count should raise ValueError."""
        ch = MIMOChannel(n_tx=4, n_rx=2, channel_type="flat")
        iq = np.zeros((2, 256), dtype=np.complex64)
        with pytest.raises(ValueError, match="TX streams"):
            ch(iq, _make_desc())

    def test_metadata_stored(self):
        """MIMO metadata should be stored in modulation_params."""
        ch = MIMOChannel(n_tx=2, n_rx=2, channel_type="flat")
        iq = np.zeros((2, 256), dtype=np.complex64)
        _, desc = ch(iq, _make_desc())
        assert desc.modulation_params["mimo"]["n_tx"] == 2
        assert desc.modulation_params["mimo"]["n_rx"] == 2

    def test_spatial_correlation(self):
        """Spatial correlation should change output statistics."""
        np.random.seed(42)
        iq = (np.random.randn(2, 1024) + 1j * np.random.randn(2, 1024)).astype(np.complex64)

        # Uncorrelated
        ch_uncorr = MIMOChannel(n_tx=2, n_rx=2, channel_type="flat")
        np.random.seed(99)
        out_uncorr, _ = ch_uncorr(iq.copy(), _make_desc())

        # Correlated RX
        R_rx = np.array([[1.0, 0.9], [0.9, 1.0]])
        ch_corr = MIMOChannel(n_tx=2, n_rx=2, channel_type="flat", spatial_correlation_rx=R_rx)
        np.random.seed(99)
        out_corr, _ = ch_corr(iq.copy(), _make_desc())

        # Outputs should differ
        assert not np.allclose(out_uncorr, out_corr)


class TestMIMOTDL:
    """TDL mode MIMO channel tests."""

    def test_tdl_shape(self):
        ch = MIMOChannel(n_tx=2, n_rx=2, channel_type="tdl", tdl_profile="TDL-A")
        iq = (np.random.randn(2, 512) + 1j * np.random.randn(2, 512)).astype(np.complex64)
        out, desc = ch(iq, _make_desc(), sample_rate=1e6)
        assert out.shape == (2, 512)
        assert out.dtype == np.complex64

    def test_invalid_channel_type(self):
        with pytest.raises(ValueError):
            MIMOChannel(channel_type="invalid")


class TestMIMODatasetIntegration:
    """NarrowbandDataset with mimo_config."""

    def test_mimo_output_shape(self):
        from spectra.datasets.narrowband import NarrowbandDataset
        from spectra.waveforms import BPSK, QPSK

        pool = [BPSK(), QPSK()]
        ds = NarrowbandDataset(
            waveform_pool=pool,
            num_samples=10,
            num_iq_samples=1024,
            sample_rate=1e6,
            mimo_config={"n_tx": 2, "n_rx": 4, "channel_type": "flat"},
            seed=42,
        )
        x, y = ds[0]
        assert x.shape == (8, 1024)  # 4 RX * 2 I/Q = 8
        assert isinstance(y, int)

    def test_backward_compat(self):
        """Without mimo_config, behavior unchanged."""
        from spectra.datasets.narrowband import NarrowbandDataset
        from spectra.waveforms import BPSK, QPSK

        pool = [BPSK(), QPSK()]
        ds = NarrowbandDataset(
            waveform_pool=pool,
            num_samples=10,
            num_iq_samples=256,
            sample_rate=1e6,
            seed=42,
        )
        x, y = ds[0]
        assert x.shape == (2, 256)
