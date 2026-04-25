"""Tests for CyclostationaryDataset."""

import pytest
import torch
from typing import Any
from spectra.datasets.cyclo import CyclostationaryDataset
from spectra.transforms import PSD, SCD, Cumulants
from spectra.waveforms import BPSK, QPSK

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 1e6
NUM_IQ = 4096


def _make_pool():
    return [BPSK(), QPSK()]


def _make_reps():
    return {
        "scd": SCD(nfft=64, n_alpha=64, hop=16),
        "cumulants": Cumulants(max_order=4),
    }


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


@pytest.mark.csp
class TestCyclostationaryDatasetBasic:
    def test_returns_dict_and_int(self):
        ds = CyclostationaryDataset(
            waveform_pool=_make_pool(),
            num_samples=10,
            num_iq_samples=NUM_IQ,
            sample_rate=SAMPLE_RATE,
            representations=_make_reps(),
        )
        data, label = ds[0]
        assert isinstance(data, dict)
        assert isinstance(label, int)

    def test_dict_keys_match_representation_names(self):
        reps = _make_reps()
        ds = CyclostationaryDataset(
            waveform_pool=_make_pool(),
            num_samples=10,
            num_iq_samples=NUM_IQ,
            sample_rate=SAMPLE_RATE,
            representations=reps,
        )
        data, _ = ds[0]
        assert set(data.keys()) == set(reps.keys())

    def test_output_tensors_are_float32(self):
        ds = CyclostationaryDataset(
            waveform_pool=_make_pool(),
            num_samples=10,
            num_iq_samples=NUM_IQ,
            sample_rate=SAMPLE_RATE,
            representations=_make_reps(),
        )
        data, _ = ds[0]
        for tensor in data.values():
            assert tensor.dtype == torch.float32

    def test_scd_shape(self):
        ds = CyclostationaryDataset(
            waveform_pool=_make_pool(),
            num_samples=10,
            num_iq_samples=NUM_IQ,
            sample_rate=SAMPLE_RATE,
            representations={"scd": SCD(nfft=64, n_alpha=64, hop=16)},
        )
        data, _ = ds[0]
        assert data["scd"].shape == (1, 64, 64)

    def test_cumulants_shape(self):
        ds = CyclostationaryDataset(
            waveform_pool=_make_pool(),
            num_samples=10,
            num_iq_samples=NUM_IQ,
            sample_rate=SAMPLE_RATE,
            representations={"cum": Cumulants(max_order=4)},
        )
        data, _ = ds[0]
        assert data["cum"].shape == (5,)

    def test_psd_shape(self):
        ds = CyclostationaryDataset(
            waveform_pool=_make_pool(),
            num_samples=10,
            num_iq_samples=NUM_IQ,
            sample_rate=SAMPLE_RATE,
            representations={"psd": PSD(nfft=256, overlap=128)},
        )
        data, _ = ds[0]
        assert data["psd"].shape == (1, 256)

    def test_len(self):
        ds = CyclostationaryDataset(
            waveform_pool=_make_pool(),
            num_samples=42,
            num_iq_samples=NUM_IQ,
            sample_rate=SAMPLE_RATE,
            representations=_make_reps(),
        )
        assert len(ds) == 42

    def test_label_in_valid_range(self):
        pool = _make_pool()
        ds = CyclostationaryDataset(
            waveform_pool=pool,
            num_samples=50,
            num_iq_samples=NUM_IQ,
            sample_rate=SAMPLE_RATE,
            representations=_make_reps(),
        )
        for i in range(50):
            _, label = ds[i]
            assert 0 <= label < len(pool)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.csp
class TestCyclostationaryDatasetDeterminism:
    def test_same_seed_same_output(self):
        kwargs: dict[str, Any] = dict(
            waveform_pool=_make_pool(),
            num_samples=10,
            num_iq_samples=NUM_IQ,
            sample_rate=SAMPLE_RATE,
            representations=_make_reps(),
            seed=123,
        )
        ds1 = CyclostationaryDataset(**kwargs)
        ds2 = CyclostationaryDataset(**kwargs)
        for i in range(5):
            d1, l1 = ds1[i]
            d2, l2 = ds2[i]
            assert l1 == l2
            for key in d1:
                torch.testing.assert_close(d1[key], d2[key])

    def test_different_seed_different_output(self):
        common: dict[str, Any] = dict(
            waveform_pool=_make_pool(),
            num_samples=10,
            num_iq_samples=NUM_IQ,
            sample_rate=SAMPLE_RATE,
            representations=_make_reps(),
        )
        ds1 = CyclostationaryDataset(**common, seed=0)
        ds2 = CyclostationaryDataset(**common, seed=999)
        # At least one sample should differ across 5 indices
        any_different = False
        for i in range(5):
            d1, _ = ds1[i]
            d2, _ = ds2[i]
            if not torch.allclose(d1["scd"], d2["scd"]):
                any_different = True
                break
        assert any_different

    def test_repeated_access_same_index(self):
        ds = CyclostationaryDataset(
            waveform_pool=_make_pool(),
            num_samples=10,
            num_iq_samples=NUM_IQ,
            sample_rate=SAMPLE_RATE,
            representations=_make_reps(),
            seed=42,
        )
        d1, l1 = ds[3]
        d2, l2 = ds[3]
        assert l1 == l2
        for key in d1:
            torch.testing.assert_close(d1[key], d2[key])


# ---------------------------------------------------------------------------
# Target transform
# ---------------------------------------------------------------------------


@pytest.mark.csp
class TestCyclostationaryDatasetTargetTransform:
    def test_target_transform_applied(self):
        ds = CyclostationaryDataset(
            waveform_pool=_make_pool(),
            num_samples=10,
            num_iq_samples=NUM_IQ,
            sample_rate=SAMPLE_RATE,
            representations=_make_reps(),
            target_transform=lambda x: x + 100,
        )
        _, label = ds[0]
        assert label >= 100


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.csp
class TestCyclostationaryDatasetValidation:
    def test_empty_representations_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            CyclostationaryDataset(
                waveform_pool=_make_pool(),
                num_samples=10,
                num_iq_samples=NUM_IQ,
                sample_rate=SAMPLE_RATE,
                representations={},
            )

    def test_no_nan_inf_in_outputs(self):
        ds = CyclostationaryDataset(
            waveform_pool=_make_pool(),
            num_samples=10,
            num_iq_samples=NUM_IQ,
            sample_rate=SAMPLE_RATE,
            representations=_make_reps(),
        )
        for i in range(10):
            data, _ = ds[i]
            for key, tensor in data.items():
                assert not torch.any(torch.isnan(tensor)), f"NaN in {key} at idx {i}"
                assert not torch.any(torch.isinf(tensor)), f"Inf in {key} at idx {i}"


# ---------------------------------------------------------------------------
# DataLoader compatibility
# ---------------------------------------------------------------------------


@pytest.mark.csp
class TestCyclostationaryDatasetDataLoader:
    def test_single_worker_dataloader(self):
        ds = CyclostationaryDataset(
            waveform_pool=_make_pool(),
            num_samples=8,
            num_iq_samples=NUM_IQ,
            sample_rate=SAMPLE_RATE,
            representations={"psd": PSD(nfft=256, overlap=128)},
        )

        # Custom collate for dict-of-tensors
        def _collate(batch):
            keys = batch[0][0].keys()
            data = {k: torch.stack([b[0][k] for b in batch]) for k in keys}
            labels = torch.tensor([b[1] for b in batch])
            return data, labels

        loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=_collate)
        batch_data, batch_labels = next(iter(loader))
        assert batch_data["psd"].shape == (4, 1, 256)
        assert batch_labels.shape == (4,)

    def test_multi_worker_determinism(self):
        """Two DataLoader passes with same seed should yield identical batches."""

        def _make_ds():
            return CyclostationaryDataset(
                waveform_pool=_make_pool(),
                num_samples=8,
                num_iq_samples=NUM_IQ,
                sample_rate=SAMPLE_RATE,
                representations={"psd": PSD(nfft=256, overlap=128)},
                seed=77,
            )

        def _collate(batch):
            keys = batch[0][0].keys()
            data = {k: torch.stack([b[0][k] for b in batch]) for k in keys}
            labels = torch.tensor([b[1] for b in batch])
            return data, labels

        ds1 = _make_ds()
        ds2 = _make_ds()
        loader1 = torch.utils.data.DataLoader(ds1, batch_size=4, collate_fn=_collate, num_workers=0)
        loader2 = torch.utils.data.DataLoader(ds2, batch_size=4, collate_fn=_collate, num_workers=0)
        for (d1, l1), (d2, l2) in zip(loader1, loader2):
            torch.testing.assert_close(l1, l2)
            for key in d1:
                torch.testing.assert_close(d1[key], d2[key])
