import os

import numpy as np
import pytest
import torch


class TestFileWriterABC:
    def test_cannot_instantiate(self):
        from spectra.utils.file_handlers.base_writer import FileWriter

        with pytest.raises(TypeError):
            FileWriter()


class TestTensorToComplex64:
    def test_2d_tensor(self):
        from spectra.utils.file_handlers.dataset_export import _tensor_to_complex64

        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # [2, 2]
        iq = _tensor_to_complex64(t)
        assert iq.dtype == np.complex64
        assert iq.shape == (2,)
        np.testing.assert_allclose(iq.real, [1.0, 2.0])
        np.testing.assert_allclose(iq.imag, [3.0, 4.0])

    def test_1d_complex_passthrough(self):
        from spectra.utils.file_handlers.dataset_export import _tensor_to_complex64

        arr = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        result = _tensor_to_complex64(arr)
        np.testing.assert_array_equal(result, arr)

    def test_2d_ndarray(self):
        from spectra.utils.file_handlers.dataset_export import _tensor_to_complex64

        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        iq = _tensor_to_complex64(arr)
        assert iq.dtype == np.complex64
        assert iq.shape == (2,)


class TestNumpyWriter:
    def test_write_creates_file(self, tmp_path):
        from spectra.utils.file_handlers.numpy_writer import NumpyWriter

        iq = np.zeros(128, dtype=np.complex64)
        path = str(tmp_path / "signal.npy")
        writer = NumpyWriter(path)
        writer.write(iq)
        assert os.path.exists(path)

    def test_data_fidelity(self, tmp_path):
        from spectra.utils.file_handlers.numpy_writer import NumpyWriter

        iq = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
        path = str(tmp_path / "signal.npy")
        NumpyWriter(path).write(iq)
        loaded = np.load(path)
        np.testing.assert_array_equal(loaded, iq)

    def test_roundtrip_with_reader(self, tmp_path):
        from spectra.utils.file_handlers.numpy_reader import NumpyReader
        from spectra.utils.file_handlers.numpy_writer import NumpyWriter

        iq = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        path = str(tmp_path / "rt.npy")
        NumpyWriter(path).write(iq)
        result, meta = NumpyReader().read(path)
        np.testing.assert_array_equal(result, iq)

    def test_extensions(self):
        from spectra.utils.file_handlers.numpy_writer import NumpyWriter

        assert NumpyWriter.extensions() == (".npy",)


class TestNumpyWriteFromDataset:
    def test_creates_folder_structure(self, tmp_path):
        from spectra.datasets import NarrowbandDataset
        from spectra.utils.file_handlers.numpy_writer import NumpyWriter
        from spectra.waveforms import BPSK, QPSK

        ds = NarrowbandDataset(
            waveform_pool=[BPSK(), QPSK()],
            num_samples=4,
            num_iq_samples=128,
            sample_rate=1e6,
            seed=42,
        )
        out = str(tmp_path / "export")
        NumpyWriter.write_from_dataset(
            ds, output_dir=out, class_list=["BPSK", "QPSK"]
        )
        npy_files = []
        for root, dirs, files in os.walk(out):
            npy_files.extend(f for f in files if f.endswith(".npy"))
        assert len(npy_files) == 4

    def test_roundtrip_to_folder_dataset(self, tmp_path):
        from spectra.datasets import NarrowbandDataset
        from spectra.datasets.folder import SignalFolderDataset
        from spectra.utils.file_handlers.numpy_writer import NumpyWriter
        from spectra.waveforms import BPSK, QPSK

        ds = NarrowbandDataset(
            waveform_pool=[BPSK(), QPSK()],
            num_samples=4,
            num_iq_samples=128,
            sample_rate=1e6,
            seed=42,
        )
        out = str(tmp_path / "export")
        NumpyWriter.write_from_dataset(
            ds, output_dir=out, class_list=["BPSK", "QPSK"]
        )
        loaded = SignalFolderDataset(root=out, num_iq_samples=128)
        assert len(loaded) == 4
        data, label = loaded[0]
        assert data.shape == (2, 128)

    def test_max_samples(self, tmp_path):
        from spectra.datasets import NarrowbandDataset
        from spectra.utils.file_handlers.numpy_writer import NumpyWriter
        from spectra.waveforms import BPSK

        ds = NarrowbandDataset(
            waveform_pool=[BPSK()],
            num_samples=10,
            num_iq_samples=64,
            sample_rate=1e6,
            seed=42,
        )
        out = str(tmp_path / "export")
        NumpyWriter.write_from_dataset(
            ds, output_dir=out, class_list=["BPSK"], max_samples=3
        )
        npy_files = []
        for root, dirs, files in os.walk(out):
            npy_files.extend(f for f in files if f.endswith(".npy"))
        assert len(npy_files) == 3


class TestRawIQWriter:
    def test_write_cf32_creates_file(self, tmp_path):
        from spectra.utils.file_handlers.raw_writer import RawIQWriter

        iq = np.zeros(128, dtype=np.complex64)
        path = str(tmp_path / "signal.cf32")
        RawIQWriter(path).write(iq)
        assert os.path.exists(path)

    def test_write_cf32_data_fidelity(self, tmp_path):
        from spectra.utils.file_handlers.raw_writer import RawIQWriter

        iq = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
        path = str(tmp_path / "signal.cf32")
        RawIQWriter(path).write(iq)
        loaded = np.fromfile(path, dtype=np.complex64)
        np.testing.assert_array_equal(loaded, iq)

    def test_write_cs16_interleaved(self, tmp_path):
        from spectra.utils.file_handlers.raw_writer import RawIQWriter

        iq = np.array([0.5 + 0.25j, -0.5 - 0.25j], dtype=np.complex64)
        path = str(tmp_path / "signal.cs16")
        RawIQWriter(path, dtype="int16").write(iq)
        raw = np.fromfile(path, dtype=np.int16)
        assert len(raw) == 4  # 2 complex samples * 2 (I + Q)

    def test_roundtrip_cf32_with_reader(self, tmp_path):
        from spectra.utils.file_handlers.raw_reader import RawIQReader
        from spectra.utils.file_handlers.raw_writer import RawIQWriter

        iq = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        path = str(tmp_path / "rt.cf32")
        RawIQWriter(path).write(iq)
        result, _ = RawIQReader().read(path)
        np.testing.assert_array_equal(result, iq)

    def test_roundtrip_cs16_with_reader(self, tmp_path):
        from spectra.utils.file_handlers.raw_reader import RawIQReader
        from spectra.utils.file_handlers.raw_writer import RawIQWriter

        iq = np.array([0.5 + 0.25j, -0.5 - 0.25j], dtype=np.complex64)
        path = str(tmp_path / "rt.cs16")
        RawIQWriter(path, dtype="int16").write(iq)
        result, _ = RawIQReader(dtype="int16").read(path)
        np.testing.assert_allclose(result.real, iq.real, atol=1e-4)
        np.testing.assert_allclose(result.imag, iq.imag, atol=1e-4)

    def test_extensions(self):
        from spectra.utils.file_handlers.raw_writer import RawIQWriter

        assert RawIQWriter.extensions() == (".cf32", ".cs16", ".raw", ".iq", ".bin")


class TestRawIQWriteFromDataset:
    def test_creates_folder_structure(self, tmp_path):
        from spectra.datasets import NarrowbandDataset
        from spectra.utils.file_handlers.raw_writer import RawIQWriter
        from spectra.waveforms import BPSK, QPSK

        ds = NarrowbandDataset(
            waveform_pool=[BPSK(), QPSK()],
            num_samples=4,
            num_iq_samples=128,
            sample_rate=1e6,
            seed=42,
        )
        out = str(tmp_path / "export")
        RawIQWriter.write_from_dataset(
            ds, output_dir=out, class_list=["BPSK", "QPSK"]
        )
        cf32_files = []
        for root, dirs, files in os.walk(out):
            cf32_files.extend(f for f in files if f.endswith(".cf32"))
        assert len(cf32_files) == 4

    def test_roundtrip_to_folder_dataset(self, tmp_path):
        from spectra.datasets import NarrowbandDataset
        from spectra.datasets.folder import SignalFolderDataset
        from spectra.utils.file_handlers.raw_writer import RawIQWriter
        from spectra.waveforms import BPSK, QPSK

        ds = NarrowbandDataset(
            waveform_pool=[BPSK(), QPSK()],
            num_samples=4,
            num_iq_samples=128,
            sample_rate=1e6,
            seed=42,
        )
        out = str(tmp_path / "export")
        RawIQWriter.write_from_dataset(
            ds, output_dir=out, class_list=["BPSK", "QPSK"]
        )
        loaded = SignalFolderDataset(root=out, num_iq_samples=128)
        assert len(loaded) == 4
        data, label = loaded[0]
        assert data.shape == (2, 128)
