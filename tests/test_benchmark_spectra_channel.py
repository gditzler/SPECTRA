import pytest
import torch


class TestLoadChannelBenchmark:
    @pytest.mark.parametrize("condition", ["awgn", "fading_los", "fading_nlos", "hardware", "full"])
    def test_condition_len(self, condition):
        from spectra.benchmarks import load_channel_benchmark
        ds = load_channel_benchmark("spectra-channel", condition=condition)
        assert len(ds) == 3_600  # 18 classes × 200 samples

    def test_sample_shape(self):
        from spectra.benchmarks import load_channel_benchmark
        ds = load_channel_benchmark("spectra-channel", condition="awgn")
        data, label = ds[0]
        assert data.shape == (2, 1024)
        assert data.dtype == torch.float32
        assert 0 <= label < 18

    def test_invalid_condition_raises(self):
        from spectra.benchmarks import load_channel_benchmark
        with pytest.raises(ValueError, match="condition"):
            load_channel_benchmark("spectra-channel", condition="bogus")

    def test_conditions_differ(self):
        from spectra.benchmarks import load_channel_benchmark
        ds_a = load_channel_benchmark("spectra-channel", condition="awgn")
        ds_f = load_channel_benchmark("spectra-channel", condition="fading_nlos")
        d_a, _ = ds_a[0]
        d_f, _ = ds_f[0]
        assert not torch.equal(d_a, d_f)

    def test_deterministic(self):
        from spectra.benchmarks import load_channel_benchmark
        ds1 = load_channel_benchmark("spectra-channel", condition="hardware")
        ds2 = load_channel_benchmark("spectra-channel", condition="hardware")
        assert ds1[42][1] == ds2[42][1]
