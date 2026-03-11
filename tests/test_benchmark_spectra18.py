import torch


class TestSpectra18Benchmark:
    def test_load_by_name(self):
        from spectra.benchmarks import load_benchmark

        ds = load_benchmark("spectra-18", split="train")
        assert len(ds) == 50_000

    def test_load_val_split(self):
        from spectra.benchmarks import load_benchmark

        ds = load_benchmark("spectra-18", split="val")
        assert len(ds) == 10_000

    def test_load_test_split(self):
        from spectra.benchmarks import load_benchmark

        ds = load_benchmark("spectra-18", split="test")
        assert len(ds) == 10_000

    def test_load_all_splits(self):
        from spectra.benchmarks import load_benchmark

        train, val, test = load_benchmark("spectra-18", split="all")
        assert len(train) == 50_000
        assert len(val) == 10_000
        assert len(test) == 10_000

    def test_sample_shape(self):
        from spectra.benchmarks import load_benchmark

        ds = load_benchmark("spectra-18", split="train")
        data, label = ds[0]
        assert data.shape == (2, 1024)
        assert data.dtype == torch.float32
        assert isinstance(label, int)
        assert 0 <= label < 18

    def test_deterministic_labels(self):
        from spectra.benchmarks import load_benchmark

        ds1 = load_benchmark("spectra-18", split="train")
        ds2 = load_benchmark("spectra-18", split="train")
        # Labels are deterministic (seeded RNG); IQ data varies due to
        # random impairments (PhaseOffset, FrequencyOffset) using global state
        for idx in [0, 42, 100]:
            _, l1 = ds1[idx]
            _, l2 = ds2[idx]
            assert l1 == l2

    def test_splits_differ(self):
        from spectra.benchmarks import load_benchmark

        train_ds = load_benchmark("spectra-18", split="train")
        val_ds = load_benchmark("spectra-18", split="val")
        d_train, _ = train_ds[0]
        d_val, _ = val_ds[0]
        assert not torch.equal(d_train, d_val)
