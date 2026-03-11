"""Tests for the speed benchmark output schema and timing helpers."""

import time as _time

import pytest


@pytest.mark.benchmark
class TestTimeInit:
    def test_returns_dataset_and_positive_time(self):
        from benchmarks.comparison.speed_benchmark import time_init

        ds, elapsed = time_init(lambda: list(range(100)))
        assert ds == list(range(100))
        assert elapsed > 0

    def test_slower_factory_produces_larger_time(self):
        from benchmarks.comparison.speed_benchmark import time_init

        _, fast = time_init(lambda: None)
        _, slow = time_init(lambda: _time.sleep(0.01))
        assert slow > fast


@pytest.mark.benchmark
class TestSpeedResultsSchema:
    def test_spectra_results_contain_init_and_amortized(self):
        """Validate the schema against a mock result dict."""
        import numpy as np

        from benchmarks.comparison.speed_benchmark import build_result_dict

        result = build_result_dict(
            init_time=0.001,
            getitem_times=np.array([0.0001] * 100),
            dl_count=100,
            dl_elapsed=0.5,
        )
        assert "init_time_s" in result
        assert "amortized_cost_ms" in result
        assert "getitem_mean_ms" in result
        assert "total_getitem_time_s" in result
        assert "num_samples_measured" in result

    def test_amortized_cost_formula(self):
        import numpy as np

        from benchmarks.comparison.speed_benchmark import build_result_dict

        times = np.array([0.001] * 200)  # 200 samples at 1ms each
        result = build_result_dict(
            init_time=0.5,
            getitem_times=times,
            dl_count=200,
            dl_elapsed=1.0,
        )
        expected = (0.5 + 0.2) / 200 * 1000  # (init + total_getitem) / n * 1000
        assert abs(result["amortized_cost_ms"] - expected) < 0.001
