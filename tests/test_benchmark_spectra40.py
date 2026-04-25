import torch


class TestSpectra40:
    def test_train_len(self):
        from spectra.benchmarks import load_benchmark

        ds = load_benchmark("spectra-40", split="train")
        assert len(ds) == 100_000

    def test_val_len(self):
        from spectra.benchmarks import load_benchmark

        assert len(load_benchmark("spectra-40", split="val")) == 20_000

    def test_test_len(self):
        from spectra.benchmarks import load_benchmark

        assert len(load_benchmark("spectra-40", split="test")) == 20_000

    def test_sample_shape(self):
        from spectra.benchmarks import load_benchmark

        ds = load_benchmark("spectra-40", split="train")
        data, label = ds[0]
        assert data.shape == (2, 1024)
        assert data.dtype == torch.float32
        assert isinstance(label, int)
        assert 0 <= label < 40

    def test_all_40_classes_present(self):
        from spectra.benchmarks import load_benchmark

        ds = load_benchmark("spectra-40", split="test")
        labels = {ds[i][1] for i in range(400)}
        assert len(labels) == 40

    def test_deterministic(self):
        from spectra.benchmarks import load_benchmark

        ds1 = load_benchmark("spectra-40", split="train")
        ds2 = load_benchmark("spectra-40", split="train")
        assert ds1[0][1] == ds2[0][1]
