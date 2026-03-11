import csv
import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from spectra.datasets.iq_utils import iq_to_tensor, truncate_pad
from spectra.utils.file_handlers.base_reader import FileReader
from spectra.utils.file_handlers.registry import get_reader


class ManifestDataset(Dataset):
    """Dataset that reads files listed in a CSV or JSON manifest.

    Useful when files are in a flat directory or when labels come from
    an external annotation file.

    **CSV format** (must have ``file`` and ``label`` columns)::

        file,label
        recording_001.sigmf-meta,BPSK
        recording_002.cf32,QPSK

    **JSON format** (list of objects)::

        [
            {"file": "recording_001.sigmf-meta", "label": "BPSK"},
            {"file": "recording_002.cf32", "label": "QPSK"}
        ]

    Args:
        manifest_path: Path to the CSV or JSON manifest file.
        root: Base directory for relative paths.  Defaults to the
            manifest file's parent directory.
        num_iq_samples: Number of IQ samples per item.
        transform: Optional callable applied to IQ array.
        target_transform: Optional callable applied to label.
        impairments: Optional impairment pipeline.
        sample_rate: Sample rate in Hz.
        reader_overrides: Optional extension-to-reader mapping.
    """

    def __init__(
        self,
        manifest_path: str,
        root: Optional[str] = None,
        num_iq_samples: int = 1024,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        impairments: Optional[Callable] = None,
        sample_rate: Optional[float] = None,
        reader_overrides: Optional[Dict[str, FileReader]] = None,
    ):
        self.manifest_path = manifest_path
        self.root = root or os.path.dirname(os.path.abspath(manifest_path))
        self.num_iq_samples = num_iq_samples
        self.transform = transform
        self.target_transform = target_transform
        self.impairments = impairments
        self.sample_rate = sample_rate
        self.reader_overrides = reader_overrides or {}

        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.samples: List[Tuple[str, int]] = []
        self._parse_manifest()

    def _parse_manifest(self) -> None:
        if self.manifest_path.endswith(".json"):
            self._parse_json()
        else:
            self._parse_csv()

        # Build sorted class list
        labels = sorted({label for _, label in self._raw_entries})
        self.classes = labels
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Resolve paths and build samples
        for filepath, label in self._raw_entries:
            if not os.path.isabs(filepath):
                filepath = os.path.join(self.root, filepath)
            self.samples.append((filepath, self.class_to_idx[label]))

    def _parse_csv(self) -> None:
        self._raw_entries: List[Tuple[str, str]] = []
        with open(self.manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._raw_entries.append((row["file"], row["label"]))

    def _parse_json(self) -> None:
        self._raw_entries: List[Tuple[str, str]] = []
        with open(self.manifest_path) as f:
            entries = json.load(f)
        for entry in entries:
            self._raw_entries.append((entry["file"], entry["label"]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, class_idx = self.samples[idx]

        # Load IQ via appropriate reader
        reader = get_reader(filepath, self.reader_overrides)
        iq, metadata = reader.read(filepath)

        # Truncate / pad to num_iq_samples
        iq = truncate_pad(iq, self.num_iq_samples)

        # Apply impairments
        if self.impairments is not None:
            from spectra.scene.signal_desc import SignalDescription

            sr = self.sample_rate or metadata.sample_rate or 1.0
            desc = SignalDescription(
                t_start=0.0,
                t_stop=self.num_iq_samples / sr,
                f_low=-sr / 2,
                f_high=sr / 2,
                label=self.classes[class_idx],
                snr=0.0,
            )
            iq, _ = self.impairments(iq, desc, sample_rate=sr)

        # Convert to tensor: [2, num_iq_samples]
        if self.transform is not None:
            data = self.transform(iq)
        else:
            data = iq_to_tensor(iq)

        label = class_idx
        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label
