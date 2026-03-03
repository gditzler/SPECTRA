import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from spectra.utils.file_handlers.base_reader import FileReader
from spectra.utils.file_handlers.registry import get_reader, supported_extensions


class SignalFolderDataset(Dataset):
    """ImageFolder-style dataset that loads RF recordings from disk.

    Expects a directory structure where each subdirectory is a class::

        root/
            BPSK/
                recording_001.sigmf-meta
                recording_002.cf32
            QPSK/
                recording_003.npy
                ...

    Directory names become class labels sorted alphabetically.
    Mixed file formats within the same directory are supported.

    Args:
        root: Path to the root directory.
        num_iq_samples: Number of IQ samples per item.  Recordings are
            truncated or zero-padded to this length.
        transform: Optional callable applied to the 1-D ``complex64``
            IQ array (e.g. :class:`Spectrogram`).
        target_transform: Optional callable applied to the integer label.
        impairments: Optional :class:`Compose` pipeline applied to IQ.
        sample_rate: Sample rate in Hz (used by impairments).
        reader_overrides: Optional mapping of file extension to
            :class:`FileReader` instance for custom reader configuration.
        extensions: Optional tuple of allowed file extensions.
    """

    def __init__(
        self,
        root: str,
        num_iq_samples: int,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        impairments: Optional[Callable] = None,
        sample_rate: Optional[float] = None,
        reader_overrides: Optional[Dict[str, FileReader]] = None,
        extensions: Optional[Tuple[str, ...]] = None,
    ):
        self.root = root
        self.num_iq_samples = num_iq_samples
        self.transform = transform
        self.target_transform = target_transform
        self.impairments = impairments
        self.sample_rate = sample_rate
        self.reader_overrides = reader_overrides or {}
        self._allowed_extensions = extensions or supported_extensions()

        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.samples: List[Tuple[str, int]] = []
        self._scan_directory()

    def _scan_directory(self) -> None:
        entries = sorted(
            e
            for e in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, e))
        )
        if not entries:
            raise FileNotFoundError(
                f"No class subdirectories found in {self.root}"
            )

        self.classes = entries
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_dir = os.path.join(self.root, cls_name)
            cls_idx = self.class_to_idx[cls_name]
            for fname in sorted(os.listdir(cls_dir)):
                _, ext = os.path.splitext(fname)
                if ext.lower() in self._allowed_extensions:
                    self.samples.append(
                        (os.path.join(cls_dir, fname), cls_idx)
                    )

        if not self.samples:
            raise FileNotFoundError(
                f"No files with supported extensions "
                f"{self._allowed_extensions} found in {self.root}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, class_idx = self.samples[idx]

        # Load IQ via appropriate reader
        reader = get_reader(filepath, self.reader_overrides)
        iq, metadata = reader.read(filepath)

        # Truncate / pad to num_iq_samples
        iq = iq[: self.num_iq_samples]
        if len(iq) < self.num_iq_samples:
            padded = np.zeros(self.num_iq_samples, dtype=np.complex64)
            padded[: len(iq)] = iq
            iq = padded

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

        # Convert to tensor: [2, num_iq_samples] (I and Q channels)
        if self.transform is not None:
            data = self.transform(iq)
        else:
            data = torch.tensor(
                np.stack([iq.real, iq.imag]), dtype=torch.float32
            )

        label = class_idx
        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label
