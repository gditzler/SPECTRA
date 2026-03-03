import pytest
from benchmarks.torchsig_compat.label_map import (
    CANONICAL_CLASSES,
    spectra_waveform_pool,
    torchsig_class_names,
    canonical_to_index,
)


def test_canonical_has_8_classes():
    assert len(CANONICAL_CLASSES) == 8


def test_spectra_pool_length_matches():
    pool = spectra_waveform_pool()
    assert len(pool) == 8


def test_spectra_pool_labels_match_canonical():
    pool = spectra_waveform_pool()
    for wf, canonical in zip(pool, CANONICAL_CLASSES):
        assert wf.label == canonical


def test_torchsig_names_length():
    names = torchsig_class_names()
    assert len(names) == 8


def test_canonical_to_index_roundtrip():
    for i, name in enumerate(CANONICAL_CLASSES):
        assert canonical_to_index(name) == i
