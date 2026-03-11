import torch


class TestSpectra18WidebandBenchmark:
    def test_load_by_name(self):
        from spectra.benchmarks import load_benchmark

        ds = load_benchmark("spectra-18-wideband", split="train")
        assert len(ds) == 20_000

    def test_load_all_splits(self):
        from spectra.benchmarks import load_benchmark

        train, val, test = load_benchmark("spectra-18-wideband", split="all")
        assert len(train) == 20_000
        assert len(val) == 4_000
        assert len(test) == 4_000

    def test_sample_returns_targets(self):
        from spectra.benchmarks import load_benchmark

        ds = load_benchmark("spectra-18-wideband", split="train")
        data, targets = ds[0]
        assert isinstance(data, torch.Tensor)
        assert isinstance(targets, dict)
        assert "boxes" in targets
        assert "labels" in targets
        assert "signal_descs" in targets

    def test_deterministic_structure(self):
        from spectra.benchmarks import load_benchmark

        ds1 = load_benchmark("spectra-18-wideband", split="train")
        ds2 = load_benchmark("spectra-18-wideband", split="train")
        _, t1 = ds1[5]
        _, t2 = ds2[5]
        # Same number of signals detected (seeded composer)
        assert len(t1["labels"]) == len(t2["labels"])
