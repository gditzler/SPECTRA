# File I/O

## Supported Formats

| Extension | Reader class | Writer class | Optional dep |
|-----------|-------------|-------------|--------------|
| `.npy` | `NumpyReader` | `NumpyWriter` | — |
| `.bin` / `.iq` | `RawIQReader` | `RawIQWriter` | — |
| `.sigmf-data` | `SigMFReader` | `SigMFWriter` | `spectra[io]` |
| `.h5` / `.hdf5` | `HDF5Reader` | `HDF5Writer` | `spectra[io]` |
| `.db` / `.sqlite` | `SQLiteReader` | `SQLiteWriter` | — |
| `.zarr` | _(ZarrHandler)_ | _(ZarrHandler)_ | `spectra[zarr]` |

---

## Auto-Detection

`get_reader()` selects the appropriate reader based on file extension:

```python
from spectra.utils.file_handlers.registry import get_reader

reader = get_reader("recordings/capture.sigmf-data")
iq = reader.read()  # -> np.ndarray[complex64]
```

You can register custom readers for new formats:

```python
from spectra.utils.file_handlers.registry import register_reader

register_reader(".custom", MyCustomReader)
```

---

## Reading Recordings

```python
from spectra.utils.file_handlers.numpy_reader import NumpyReader
from spectra.utils.file_handlers.sigmf_reader import SigMFReader

# NumPy
reader = NumpyReader("capture.npy")
iq = reader.read()

# SigMF (requires spectra[io])
reader = SigMFReader("capture.sigmf-data")
iq = reader.read()
metadata = reader.metadata  # dict with SigMF annotations
```

---

## Writing Datasets

### Export to folder structure

```python
from spectra import NarrowbandDataset, QPSK, BPSK, AWGN, Compose
from spectra.utils.file_handlers.dataset_export import export_dataset_to_folder

dataset = NarrowbandDataset(
    waveform_pool=[QPSK(), BPSK()],
    num_samples=1000,
    num_iq_samples=1024,
    sample_rate=1e6,
    impairments=Compose([AWGN(snr_range=(5.0, 20.0))]),
    seed=42,
)

export_dataset_to_folder(
    dataset=dataset,
    output_dir="exported_dataset/",
    format="npy",          # or "sigmf", "hdf5", "raw"
    num_workers=4,
)
# Creates: exported_dataset/QPSK/000000.npy, exported_dataset/BPSK/000001.npy, ...
```

### Write individual files

```python
from spectra.utils.file_handlers.sigmf_writer import SigMFWriter

writer = SigMFWriter("output.sigmf-data", sample_rate=1e6, center_freq=0.0)
writer.write(iq)
```

---

## Optional Dependencies

!!! note "HDF5 and SigMF require extras"
    - HDF5: `pip install 'spectra[io]'` (installs `h5py`)
    - SigMF: `pip install 'spectra[io]'` (installs `sigmf`)
    - Zarr: `pip install 'spectra[zarr]'` (installs `zarr`)

    Attempting to use these without the extras raises `ImportError` with a
    clear message indicating which extra to install.
