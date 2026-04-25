from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from spectra.scene.signal_desc import SignalDescription


@dataclass
class STFTParams:
    nfft: int
    hop_length: int
    sample_rate: float
    num_samples: int

    @property
    def num_time_bins(self) -> int:
        return (self.num_samples - self.nfft) // self.hop_length + 1

    @property
    def num_freq_bins(self) -> int:
        return self.nfft

    @property
    def freq_resolution(self) -> float:
        return self.sample_rate / self.nfft

    @property
    def time_resolution(self) -> float:
        return self.hop_length / self.sample_rate


def to_coco(
    signal_descs: List[SignalDescription],
    stft_params: STFTParams,
    class_list: List[str],
) -> Dict[str, Any]:
    if len(signal_descs) == 0:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "signal_descs": [],
        }

    boxes = []
    labels = []
    p = stft_params

    for desc in signal_descs:
        # Time -> pixel coords (x axis)
        x_min = desc.t_start / p.time_resolution
        x_max = desc.t_stop / p.time_resolution

        # Frequency -> pixel coords (y axis)
        # After fftshift, frequency axis is [-fs/2, fs/2] mapped to [0, nfft]
        half_fs = p.sample_rate / 2.0
        y_min = (desc.f_low + half_fs) / p.sample_rate * p.nfft
        y_max = (desc.f_high + half_fs) / p.sample_rate * p.nfft

        # Clamp to spectrogram bounds
        x_min = max(0.0, x_min)
        x_max = min(float(p.num_time_bins), x_max)
        y_min = max(0.0, y_min)
        y_max = min(float(p.num_freq_bins), y_max)

        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(class_list.index(desc.label))

    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "signal_descs": signal_descs,
    }
