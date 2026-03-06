# Installation

## Prerequisites

- Python 3.10 or newer
- Rust 1.83 or newer ([install Rust](https://rustup.rs/))

## Install with uv (recommended)

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install maturin

# CPU-only PyTorch (avoids downloading large CUDA builds)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Build and install SPECTRA (compiles the Rust extension)
maturin develop --release
```

## Install with pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin
pip install torch --index-url https://download.pytorch.org/whl/cpu
maturin develop --release
```

## Optional extras

| Extra | Installs | Use case |
|-------|----------|----------|
| `spectra[classifiers]` | scikit-learn | `CyclostationaryAMC` |
| `spectra[io]` | sigmf, h5py | SigMF and HDF5 file I/O |
| `spectra[zarr]` | zarr | Zarr dataset export |
| `spectra[all]` | everything above | Full feature set |
| `spectra[docs]` | mkdocs, material | Build this documentation |

## Verify the install

```python
import spectra
from spectra import QPSK, AWGN
print("SPECTRA installed successfully")
```

!!! warning "After Rust changes"
    Any time you modify files under `rust/`, re-run `maturin develop --release`
    before running Python code or building the docs.
