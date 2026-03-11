from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from spectra.impairments.compose import Compose
from spectra.scene.composer import Composer, SceneConfig
from spectra.scene.labels import STFTParams, to_coco


class WidebandDataset(Dataset):
    def __init__(
        self,
        scene_config: SceneConfig,
        num_samples: int,
        impairments: Optional[Compose] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ):
        self.scene_config = scene_config
        self.num_samples = num_samples
        self.impairments = impairments
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed if seed is not None else 0
        self.composer = Composer(scene_config)

        # Build class list from signal pool
        self.class_list = sorted(set(w.label for w in scene_config.signal_pool))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        # Deterministic seed from (base_seed, idx)
        rng = np.random.default_rng(seed=(self.seed, idx))
        scene_seed = int(rng.integers(0, 2**32))

        iq, signal_descs = self.composer.generate(seed=scene_seed)

        # Apply scene-level impairments
        if self.impairments is not None:
            from spectra.scene.signal_desc import SignalDescription

            scene_desc = SignalDescription(
                t_start=0.0,
                t_stop=self.scene_config.capture_duration,
                f_low=-self.scene_config.capture_bandwidth / 2,
                f_high=self.scene_config.capture_bandwidth / 2,
                label="scene",
                snr=0.0,
            )
            iq, _ = self.impairments(iq, scene_desc, sample_rate=self.scene_config.sample_rate)

        # Apply transform (e.g., STFT)
        if self.transform is not None:
            data = self.transform(iq)
            # Build STFT params for label conversion
            stft = self.transform
            stft_params = STFTParams(
                nfft=stft.nfft,
                hop_length=stft.hop_length,
                sample_rate=self.scene_config.sample_rate,
                num_samples=len(iq),
            )
            targets = to_coco(signal_descs, stft_params, self.class_list)
        else:
            data = torch.tensor(np.stack([iq.real, iq.imag]), dtype=torch.float32)
            targets = {
                "boxes": torch.zeros((0, 4)),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "signal_descs": signal_descs,
            }

        targets["signal_descs"] = signal_descs

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return data, targets
