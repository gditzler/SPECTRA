"""
Dataset I/O and Export
======================
Level: Intermediate

Demonstrate SPECTRA's dataset writing and export utilities:
  - DatasetWriter — batch-generate and save a dataset
  - export_dataset_to_folder — export to class-per-directory structure
  - NumpyWriter — write individual IQ files as .npy

Run:
    python examples/datasets/dataset_io.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.waveforms import BPSK, QPSK, QAM16
from spectra.impairments import AWGN
from spectra.datasets import NarrowbandDataset
from spectra.utils.file_handlers import NumpyWriter
from spectra.utils.file_handlers.dataset_export import export_dataset_to_folder
from plot_helpers import savefig

sample_rate = 1e6

# ── 1. Build a small dataset ────────────────────────────────────────────────
dataset = NarrowbandDataset(
    waveform_pool=[BPSK(), QPSK(), QAM16()],
    num_samples=60,  # 3 classes × 20
    num_iq_samples=1024,
    sample_rate=sample_rate,
    impairments=AWGN(snr=15.0),
    seed=42,
)
print(f"Built dataset: {len(dataset)} samples")

# ── 2. Export to folder structure ────────────────────────────────────────────
tmpdir = tempfile.mkdtemp(prefix="spectra_export_")
export_dir = Path(tmpdir) / "exported"

export_dataset_to_folder(
    dataset=dataset,
    output_dir=str(export_dir),
    writer_factory=lambda path: NumpyWriter(path),
    file_extension=".npy",
)

# List exported structure
class_dirs = sorted(p for p in export_dir.iterdir() if p.is_dir())
print(f"\nExported to: {export_dir}")
for d in class_dirs:
    files = list(d.glob("*.npy"))
    print(f"  {d.name}/: {len(files)} files")

# ── 3. Verify by loading back ────────────────────────────────────────────────
sample_file = list(class_dirs[0].glob("*.npy"))[0]
iq_loaded = np.load(sample_file)
print(f"\nLoaded back: {sample_file.name}, shape={iq_loaded.shape}, dtype={iq_loaded.dtype}")

# ── 4. Visualize original vs loaded ─────────────────────────────────────────
iq_original, _ = dataset[0]

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(iq_original[:200].real, linewidth=0.5)
axes[0].set_title("Original (from dataset)")
axes[0].grid(True, alpha=0.3)
axes[1].plot(iq_loaded[:200].real, linewidth=0.5, color="tab:orange")
axes[1].set_title("Loaded (from .npy export)")
axes[1].grid(True, alpha=0.3)
fig.suptitle("Dataset Export & Reload Verification", fontsize=12)
fig.tight_layout()
savefig("dataset_io_verification.png")
plt.close()

print(f"\nDone — dataset I/O example saved. Temp dir: {tmpdir}")
