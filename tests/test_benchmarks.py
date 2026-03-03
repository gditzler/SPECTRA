import os
import tempfile

import pytest
import torch


class TestLoadBenchmark:
    def test_load_from_yaml_file_narrowband(self, tmp_path):
        from spectra.benchmarks import load_benchmark

        config_content = """
name: "test-nb"
version: "1.0"
description: "Test narrowband benchmark"
task: "narrowband"
sample_rate: 1000000
num_iq_samples: 256
num_samples:
  train: 16
  val: 8
  test: 8
seed:
  train: 100
  val: 200
  test: 300
waveform_pool:
  - {type: "QPSK"}
  - {type: "BPSK"}
snr_range: [0, 20]
impairments:
  - {type: "AWGN"}
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(config_content)

        train_ds = load_benchmark(str(config_file), split="train")
        assert len(train_ds) == 16
        data, label = train_ds[0]
        assert isinstance(data, torch.Tensor)
        assert data.shape == (2, 256)
        assert isinstance(label, int)

    def test_load_all_splits(self, tmp_path):
        from spectra.benchmarks import load_benchmark

        config_content = """
name: "test-splits"
version: "1.0"
task: "narrowband"
sample_rate: 1000000
num_iq_samples: 256
num_samples:
  train: 16
  val: 8
  test: 8
seed:
  train: 100
  val: 200
  test: 300
waveform_pool:
  - {type: "QPSK"}
snr_range: [0, 20]
impairments: []
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(config_content)

        train_ds, val_ds, test_ds = load_benchmark(str(config_file), split="all")
        assert len(train_ds) == 16
        assert len(val_ds) == 8
        assert len(test_ds) == 8

    def test_different_splits_have_different_seeds(self, tmp_path):
        from spectra.benchmarks import load_benchmark

        config_content = """
name: "test-seeds"
version: "1.0"
task: "narrowband"
sample_rate: 1000000
num_iq_samples: 256
num_samples:
  train: 8
  val: 8
  test: 8
seed:
  train: 100
  val: 200
  test: 300
waveform_pool:
  - {type: "QPSK"}
snr_range: [0, 20]
impairments: []
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(config_content)

        train_ds, val_ds, _ = load_benchmark(str(config_file), split="all")
        d_train, _ = train_ds[0]
        d_val, _ = val_ds[0]
        assert not torch.equal(d_train, d_val)

    def test_waveform_with_params(self, tmp_path):
        from spectra.benchmarks import load_benchmark

        config_content = """
name: "test-params"
version: "1.0"
task: "narrowband"
sample_rate: 1000000
num_iq_samples: 256
num_samples:
  train: 8
  val: 4
  test: 4
seed:
  train: 100
  val: 200
  test: 300
waveform_pool:
  - {type: "FSK", params: {order: 4}}
  - {type: "QPSK", params: {samples_per_symbol: 4}}
snr_range: [5, 25]
impairments: []
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(config_content)

        ds = load_benchmark(str(config_file), split="train")
        data, label = ds[0]
        assert data.shape == (2, 256)

    def test_wideband_task(self, tmp_path):
        from spectra.benchmarks import load_benchmark

        config_content = """
name: "test-wb"
version: "1.0"
task: "wideband"
sample_rate: 2000000
num_iq_samples: 1024
num_samples:
  train: 8
  val: 4
  test: 4
seed:
  train: 100
  val: 200
  test: 300
waveform_pool:
  - {type: "QPSK"}
  - {type: "BPSK"}
snr_range: [5, 25]
impairments: []
scene:
  capture_bandwidth: 1000000
  capture_duration: 0.001
  num_signals: [1, 3]
  allow_overlap: true
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(config_content)

        ds = load_benchmark(str(config_file), split="train")
        data, targets = ds[0]
        assert isinstance(data, torch.Tensor)
        assert isinstance(targets, dict)
        assert "signal_descs" in targets

    def test_invalid_file_raises(self):
        from spectra.benchmarks import load_benchmark

        with pytest.raises(FileNotFoundError):
            load_benchmark("/nonexistent/path.yaml", split="train")

    def test_invalid_split_raises(self, tmp_path):
        from spectra.benchmarks import load_benchmark

        config_content = """
name: "test"
version: "1.0"
task: "narrowband"
sample_rate: 1000000
num_iq_samples: 256
num_samples: {train: 8, val: 4, test: 4}
seed: {train: 1, val: 2, test: 3}
waveform_pool: [{type: "QPSK"}]
snr_range: [0, 20]
impairments: []
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(config_content)

        with pytest.raises(ValueError, match="split"):
            load_benchmark(str(config_file), split="invalid")
