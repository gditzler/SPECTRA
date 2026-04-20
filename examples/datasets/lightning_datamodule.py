"""
SPECTRA Example: PyTorch Lightning DataModules
==============================================
Level: Advanced

Wrap SPECTRA datasets in ``LightningDataModule`` so that Lightning ``Trainer``
owns the train/val/test/predict loaders and their DataLoader configuration. See
https://lightning.ai/docs/pytorch/stable/data/datamodule.html for the DataModule
lifecycle (``prepare_data`` → ``setup`` → ``*_dataloader``).

Covers:
- A narrowband DataModule for modulation classification
  (``NarrowbandDataset`` → class-index targets)
- A wideband DataModule for detection-style scenes
  (``WidebandDataset`` + ``collate_fn`` for variable-length boxes)
- Optional lightweight ``LightningModule`` + ``Trainer.fit`` smoke test

Install Lightning with::

    pip install lightning
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

try:
    import lightning as L
except ImportError:  # pragma: no cover - fallback for older installs
    try:
        import pytorch_lightning as L  # type: ignore[no-redef]
    except ImportError as exc:
        raise SystemExit(
            "This example requires Lightning. Install with: pip install lightning"
        ) from exc

import spectra as sp


# ── 1. Narrowband DataModule ────────────────────────────────────────────────
# NarrowbandDataset.__getitem__ returns (iq_tensor, class_index). The integer
# label is already what a classification head expects, so no target_transform
# is needed. We split a single base dataset into train/val/test by index so
# that every partition sees the same deterministic (seed, idx) seeding.


@dataclass
class NarrowbandSplits:
    """Fractional sizes for train/val/test splits (must sum to 1.0)."""

    train: float = 0.7
    val: float = 0.15
    test: float = 0.15


class NarrowbandDataModule(L.LightningDataModule):
    """LightningDataModule for single-signal AMC training."""

    def __init__(
        self,
        waveform_pool: list,
        num_samples: int = 800,
        num_iq_samples: int = 1024,
        sample_rate: float = 1e6,
        snr_range: tuple[float, float] = (5.0, 25.0),
        splits: NarrowbandSplits = NarrowbandSplits(),
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["waveform_pool"])
        self.waveform_pool = waveform_pool
        self.num_samples = num_samples
        self.num_iq_samples = num_iq_samples
        self.sample_rate = sample_rate
        self.snr_range = snr_range
        self.splits = splits
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.class_names = [w.label for w in waveform_pool]
        self.num_classes = len(self.class_names)

        self._train: Optional[Subset] = None
        self._val: Optional[Subset] = None
        self._test: Optional[Subset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        impairments = sp.Compose(
            [
                sp.AWGN(snr_range=self.snr_range),
                sp.FrequencyOffset(max_offset=1_000.0),
            ]
        )
        base = sp.NarrowbandDataset(
            waveform_pool=self.waveform_pool,
            num_samples=self.num_samples,
            num_iq_samples=self.num_iq_samples,
            sample_rate=self.sample_rate,
            impairments=impairments,
            transform=sp.ComplexTo2D(),
            seed=self.seed,
        )

        n = len(base)
        n_train = int(n * self.splits.train)
        n_val = int(n * self.splits.val)
        n_test = n - n_train - n_val

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        train_idx = perm[:n_train].tolist()
        val_idx = perm[n_train : n_train + n_val].tolist()
        test_idx = perm[n_train + n_val : n_train + n_val + n_test].tolist()

        self._train = Subset(base, train_idx)
        self._val = Subset(base, val_idx)
        self._test = Subset(base, test_idx)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


# ── 2. Wideband DataModule ──────────────────────────────────────────────────
# WidebandDataset returns (spectrogram_tensor, target_dict) where target_dict
# holds "boxes", "labels", etc. Because box counts vary between scenes, we use
# spectra.collate_fn so targets stay as a list rather than being stacked.


class WidebandDataModule(L.LightningDataModule):
    """LightningDataModule for multi-signal detection scenes."""

    def __init__(
        self,
        scene_config: sp.SceneConfig,
        num_samples: int = 128,
        val_samples: int = 32,
        test_samples: int = 32,
        batch_size: int = 4,
        num_workers: int = 0,
        transform=None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["scene_config", "transform"])
        self.scene_config = scene_config
        self.num_samples = num_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform or sp.STFT(nfft=512, hop_length=128)
        self.seed = seed

        self._train: Optional[sp.WidebandDataset] = None
        self._val: Optional[sp.WidebandDataset] = None
        self._test: Optional[sp.WidebandDataset] = None

    def _make(self, num: int, seed: int) -> sp.WidebandDataset:
        return sp.WidebandDataset(
            scene_config=self.scene_config,
            num_samples=num,
            transform=self.transform,
            seed=seed,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        self._train = self._make(self.num_samples, self.seed)
        self._val = self._make(self.val_samples, self.seed + 1)
        self._test = self._make(self.test_samples, self.seed + 2)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=sp.collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=sp.collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=sp.collate_fn,
            persistent_workers=self.num_workers > 0,
        )


# ── 3. Tiny classifier LightningModule for the narrowband smoke test ────────


class TinyAMC(L.LightningModule):
    """Small 1-D CNN classifier just large enough to demonstrate the loop."""

    def __init__(self, num_classes: int, num_iq_samples: int = 1024) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(2, 16, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(4),
            torch.nn.Conv1d(16, 32, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(32, num_classes),
        )
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x.float())
        loss = self.loss(logits, y.long())
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# ── 4. Driver: exercise both DataModules ────────────────────────────────────


def _demo_narrowband() -> None:
    print("── Narrowband DataModule ──")
    waveform_pool = [
        sp.BPSK(),
        sp.QPSK(),
        sp.PSK8(),
        sp.QAM16(),
        sp.FSK(),
        sp.OFDM(),
    ]
    dm = NarrowbandDataModule(
        waveform_pool=waveform_pool,
        num_samples=600,
        num_iq_samples=1024,
        batch_size=32,
        seed=42,
    )
    dm.setup()
    print(f"Classes ({dm.num_classes}): {dm.class_names}")
    print(
        f"Split sizes: train={len(dm._train)}  "
        f"val={len(dm._val)}  test={len(dm._test)}"
    )

    x, y = next(iter(dm.train_dataloader()))
    print(f"Train batch: x={tuple(x.shape)}  y={tuple(y.shape)}  dtype={x.dtype}")

    model = TinyAMC(num_classes=dm.num_classes, num_iq_samples=1024)
    trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=4,
        limit_val_batches=2,
        enable_checkpointing=False,
        logger=False,
        accelerator="cpu",
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, verbose=False)


def _demo_wideband() -> None:
    print("\n── Wideband DataModule ──")
    scene_config = sp.SceneConfig(
        capture_duration=1e-3,
        capture_bandwidth=10e6,
        sample_rate=10e6,
        num_signals=(2, 5),
        signal_pool=[sp.QPSK(), sp.QAM16(), sp.FSK(), sp.OFDM(), sp.BPSK()],
        snr_range=(5.0, 25.0),
        allow_overlap=True,
    )
    dm = WidebandDataModule(
        scene_config=scene_config,
        num_samples=16,
        val_samples=8,
        test_samples=8,
        batch_size=4,
        seed=42,
    )
    dm.setup()

    x, targets = next(iter(dm.train_dataloader()))
    print(f"Train batch spectrogram: x={tuple(x.shape)}")
    print(f"Targets is a list of dicts (len={len(targets)}); first sample:")
    first = targets[0]
    for k, v in first.items():
        shape = tuple(v.shape) if hasattr(v, "shape") else v
        print(f"  {k}: {shape}")
    print(
        f"Split sizes: train={len(dm._train)}  "
        f"val={len(dm._val)}  test={len(dm._test)}"
    )


def main() -> None:
    torch.manual_seed(0)
    _demo_narrowband()
    _demo_wideband()
    print("\nDone.")


if __name__ == "__main__":
    main()
