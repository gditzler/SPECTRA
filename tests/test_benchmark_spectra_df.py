import pytest
import torch


@pytest.mark.benchmark
class TestSpectraDf:
    def test_train_len(self):
        from spectra.benchmarks import load_benchmark

        ds = load_benchmark("spectra-df", split="train")
        assert len(ds) == 20_000

    def test_val_test_len(self):
        from spectra.benchmarks import load_benchmark

        assert len(load_benchmark("spectra-df", split="val")) == 4_000
        assert len(load_benchmark("spectra-df", split="test")) == 4_000

    def test_sample_shape(self):
        from spectra.benchmarks import load_benchmark
        from spectra.datasets import DirectionFindingTarget

        ds = load_benchmark("spectra-df", split="train")
        data, target = ds[0]
        assert data.shape == (8, 2, 256)
        assert data.dtype == torch.float32
        assert isinstance(target, DirectionFindingTarget)
        assert target.num_sources == 2

    def test_all_split(self):
        from spectra.benchmarks import load_benchmark

        train, val, test = load_benchmark("spectra-df", split="all")
        assert len(train) == 20_000
        assert len(val) == 4_000
        assert len(test) == 4_000

    def test_deterministic(self):
        from spectra.benchmarks import load_benchmark

        ds1 = load_benchmark("spectra-df", split="train")
        ds2 = load_benchmark("spectra-df", split="train")
        assert torch.equal(ds1[0][0], ds2[0][0])
