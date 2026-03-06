import torch
import pytest


def _make_ds(snr_levels=None, samples_per_cell=5, seed=42):
    from spectra.datasets.snr_sweep import SNRSweepDataset
    from spectra.waveforms import BPSK, QPSK
    from spectra.impairments import AWGN, Compose
    snr_levels = snr_levels or [-10.0, 0.0, 10.0]
    def imps_fn(snr_db):
        return Compose([AWGN(snr=snr_db)])
    return SNRSweepDataset(
        waveform_pool=[BPSK(), QPSK()],
        snr_levels=snr_levels,
        samples_per_cell=samples_per_cell,
        num_iq_samples=256,
        sample_rate=1_000_000,
        impairments_fn=imps_fn,
        seed=seed,
    )


class TestSNRSweepDataset:
    def test_length(self):
        # S=3, C=2, K=5 → 30
        assert len(_make_ds()) == 30

    def test_returns_triple(self):
        data, label, snr = _make_ds()[0]
        assert isinstance(data, torch.Tensor)
        assert isinstance(label, int)
        assert isinstance(snr, float)

    def test_tensor_shape(self):
        data, _, _ = _make_ds()[0]
        assert data.shape == (2, 256)
        assert data.dtype == torch.float32

    def test_snr_grouping(self):
        # First 10 → SNR=-10, next 10 → SNR=0, last 10 → SNR=10
        ds = _make_ds(snr_levels=[-10.0, 0.0, 10.0], samples_per_cell=5)
        for i in range(10):
            assert ds[i][2] == -10.0
        for i in range(10, 20):
            assert ds[i][2] == 0.0

    def test_class_ordering_within_snr(self):
        # idx 0,1,2 → class 0; idx 3,4,5 → class 1
        ds = _make_ds(snr_levels=[5.0], samples_per_cell=3)
        for i in range(3):
            assert ds[i][1] == 0
        for i in range(3, 6):
            assert ds[i][1] == 1

    def test_deterministic(self):
        ds1, ds2 = _make_ds(seed=99), _make_ds(seed=99)
        d1, l1, s1 = ds1[7]
        d2, l2, s2 = ds2[7]
        assert torch.equal(d1, d2) and l1 == l2 and s1 == s2
