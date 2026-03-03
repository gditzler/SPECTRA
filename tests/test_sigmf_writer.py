import json
import os

import numpy as np
import pytest


class TestSigMFWriter:
    def test_write_creates_files(self, tmp_path):
        from spectra.utils.file_handlers.sigmf_writer import SigMFWriter

        iq = np.zeros(128, dtype=np.complex64)
        base = str(tmp_path / "signal")
        writer = SigMFWriter(base_path=base, sample_rate=1e6)
        writer.write(iq)
        assert os.path.exists(base + ".sigmf-meta")
        assert os.path.exists(base + ".sigmf-data")

    def test_metadata_content(self, tmp_path):
        from spectra.utils.file_handlers.sigmf_writer import SigMFWriter

        iq = np.zeros(64, dtype=np.complex64)
        base = str(tmp_path / "signal")
        writer = SigMFWriter(
            base_path=base, sample_rate=2e6, center_frequency=915e6
        )
        writer.write(iq)
        with open(base + ".sigmf-meta") as f:
            meta = json.load(f)
        assert meta["global"]["core:datatype"] == "cf32_le"
        assert meta["global"]["core:sample_rate"] == 2e6
        assert meta["captures"][0]["core:frequency"] == 915e6

    def test_roundtrip(self, tmp_path):
        from spectra.utils.file_handlers.sigmf_reader import SigMFReader
        from spectra.utils.file_handlers.sigmf_writer import SigMFWriter

        iq = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
        base = str(tmp_path / "rt")
        SigMFWriter(base_path=base, sample_rate=1e6).write(iq)
        result, meta = SigMFReader().read(base + ".sigmf-meta")
        np.testing.assert_array_equal(result, iq)

    def test_annotations(self, tmp_path):
        from spectra.utils.file_handlers.sigmf_writer import SigMFWriter

        iq = np.zeros(256, dtype=np.complex64)
        base = str(tmp_path / "ann")
        annotations = [
            {
                "core:sample_start": 0,
                "core:sample_count": 128,
                "core:description": "BPSK",
            }
        ]
        SigMFWriter(base_path=base, sample_rate=1e6).write(
            iq, annotations=annotations
        )
        with open(base + ".sigmf-meta") as f:
            meta = json.load(f)
        assert len(meta["annotations"]) == 1
        assert meta["annotations"][0]["core:description"] == "BPSK"


class TestSigMFWriteFromDataset:
    def test_creates_folder_structure(self, tmp_path):
        from spectra.datasets import NarrowbandDataset
        from spectra.utils.file_handlers.sigmf_writer import SigMFWriter
        from spectra.waveforms import BPSK, QPSK

        ds = NarrowbandDataset(
            waveform_pool=[BPSK(), QPSK()],
            num_samples=4,
            num_iq_samples=128,
            sample_rate=1e6,
            seed=42,
        )
        out = str(tmp_path / "export")
        SigMFWriter.write_from_dataset(
            ds,
            output_dir=out,
            sample_rate=1e6,
            class_list=["BPSK", "QPSK"],
        )
        assert os.path.isdir(out)
        sigmf_files = []
        for root, dirs, files in os.walk(out):
            sigmf_files.extend(f for f in files if f.endswith(".sigmf-meta"))
        assert len(sigmf_files) == 4

    def test_roundtrip_to_folder_dataset(self, tmp_path):
        from spectra.datasets import NarrowbandDataset
        from spectra.datasets.folder import SignalFolderDataset
        from spectra.utils.file_handlers.sigmf_writer import SigMFWriter
        from spectra.waveforms import BPSK, QPSK

        ds = NarrowbandDataset(
            waveform_pool=[BPSK(), QPSK()],
            num_samples=4,
            num_iq_samples=128,
            sample_rate=1e6,
            seed=42,
        )
        out = str(tmp_path / "export")
        SigMFWriter.write_from_dataset(
            ds,
            output_dir=out,
            sample_rate=1e6,
            class_list=["BPSK", "QPSK"],
        )
        loaded = SignalFolderDataset(root=out, num_iq_samples=128)
        assert len(loaded) == 4
        data, label = loaded[0]
        assert data.shape == (2, 128)
