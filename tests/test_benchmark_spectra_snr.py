import pytest
import torch


class TestLoadSNRSweep:
    def test_test_split_len(self):
        from spectra.benchmarks import load_snr_sweep
        ds = load_snr_sweep("spectra-snr", split="test")
        # 26 SNR levels × 18 classes × 50 = 23,400
        assert len(ds) == 23_400

    def test_train_split_len(self):
        from spectra.benchmarks import load_snr_sweep
        ds = load_snr_sweep("spectra-snr", split="train")
        # 26 × 18 × 200 = 93,600
        assert len(ds) == 93_600

    def test_returns_triple(self):
        from spectra.benchmarks import load_snr_sweep
        ds = load_snr_sweep("spectra-snr", split="test")
        data, label, snr = ds[0]
        assert data.shape == (2, 1024)
        assert isinstance(label, int)
        assert isinstance(snr, float)

    def test_snr_range_coverage(self):
        from spectra.benchmarks import load_snr_sweep
        ds = load_snr_sweep("spectra-snr", split="test")
        stride = 18 * 50  # samples per SNR level
        observed = {ds[i * stride][2] for i in range(26)}
        expected = {float(v) for v in range(-20, 32, 2)}
        assert observed == expected

    def test_invalid_split_raises(self):
        from spectra.benchmarks import load_snr_sweep
        with pytest.raises(ValueError):
            load_snr_sweep("spectra-snr", split="all")

    def test_wrong_task_raises(self):
        from spectra.benchmarks import load_snr_sweep
        with pytest.raises(ValueError, match="task"):
            load_snr_sweep("spectra-18", split="test")  # wrong task


class TestEvaluateSNRSweep:
    @pytest.mark.slow
    def test_returns_all_snr_levels(self):
        from spectra.benchmarks import load_snr_sweep, evaluate_snr_sweep
        ds = load_snr_sweep("spectra-snr", split="test")
        def predict(batch): return torch.zeros(batch.shape[0], dtype=torch.long)
        results = evaluate_snr_sweep(predict, ds)
        assert len(results) == 26
        assert all("accuracy" in v and "per_class" in v for v in results.values())

    @pytest.mark.slow
    def test_accuracy_in_range(self):
        from spectra.benchmarks import load_snr_sweep, evaluate_snr_sweep
        ds = load_snr_sweep("spectra-snr", split="test")
        def predict(batch): return torch.zeros(batch.shape[0], dtype=torch.long)
        results = evaluate_snr_sweep(predict, ds)
        for snr, metrics in results.items():
            assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_wrong_dataset_type_raises(self):
        from spectra.benchmarks import load_benchmark, evaluate_snr_sweep
        ds = load_benchmark("spectra-18", split="test")
        with pytest.raises(TypeError):
            evaluate_snr_sweep(lambda x: x, ds)
