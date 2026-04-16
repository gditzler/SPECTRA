"""
Streaming DataLoader with Curriculum Learning
==============================================
Level: Advanced

Demonstrate epoch-aware data generation with difficulty progression:
  - CurriculumSchedule — linearly interpolate SNR over training
  - StreamingDataLoader — fresh deterministic DataLoader per epoch

Run:
    python examples/datasets/streaming_curriculum.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.waveforms import BPSK, QPSK, QAM16, FSK
from spectra.impairments import AWGN
from spectra.datasets import NarrowbandDataset
from spectra.curriculum import CurriculumSchedule
from spectra.streaming import StreamingDataLoader
from plot_helpers import savefig

sample_rate = 1e6

# ── 1. Define curriculum schedule ────────────────────────────────────────────
curriculum = CurriculumSchedule(
    snr_range={"start": (20.0, 20.0), "end": (0.0, 0.0)},  # start easy (20 dB) → hard (0 dB)
)

print("Curriculum progression:")
for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
    params = curriculum.at(progress)
    print(f"  progress={progress:.0%}: {params}")


# ── 2. Dataset factory ──────────────────────────────────────────────────────
def make_dataset(params):
    snr_range = params.get("snr_range", (15.0, 15.0))
    snr = float(snr_range[0])  # use the low end of the current range
    return NarrowbandDataset(
        waveform_pool=[BPSK(), QPSK(), QAM16(), FSK()],
        num_samples=100,  # 4 classes × 25
        num_iq_samples=1024,
        sample_rate=sample_rate,
        impairments=AWGN(snr=snr),
        seed=0,  # StreamingDataLoader overrides this per epoch
    )


# ── 3. Create StreamingDataLoader ────────────────────────────────────────────
num_epochs = 5
loader = StreamingDataLoader(
    dataset_factory=make_dataset,
    base_seed=42,
    num_epochs=num_epochs,
    curriculum=curriculum,
    batch_size=16,
    num_workers=0,
)

# ── 4. Iterate and collect stats ─────────────────────────────────────────────
epoch_snrs = []
epoch_powers = []

for epoch_idx in range(num_epochs):
    dl = loader.epoch(epoch_idx)
    batch_powers = []
    for batch_iq, batch_labels in dl:
        power = (batch_iq.abs() ** 2).mean().item()
        batch_powers.append(power)

    progress = epoch_idx / max(num_epochs - 1, 1)
    params = curriculum.at(progress)
    epoch_snrs.append(float(params.get("snr_range", (15.0, 15.0))[0]))
    epoch_powers.append(np.mean(batch_powers))
    print(f"Epoch {epoch_idx}: SNR={epoch_snrs[-1]:.1f} dB, mean power={epoch_powers[-1]:.4f}")

# ── 5. Plot curriculum progression ───────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(range(num_epochs), epoch_snrs, "o-", color="tab:blue", label="SNR (dB)")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("SNR (dB)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(range(num_epochs), epoch_powers, "s--", color="tab:orange", label="Mean Power")
ax2.set_ylabel("Mean Power", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")

fig.suptitle("Curriculum Learning: SNR Progression Over Epochs", fontsize=12)
fig.tight_layout()
savefig("streaming_curriculum.png")
plt.close()

print("Done — streaming/curriculum example saved.")
