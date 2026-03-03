"""Canonical 8-class label mapping between SPECTRA and TorchSig."""
import spectra as sp

CANONICAL_CLASSES = [
    "BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "OOK", "FSK", "OFDM",
]

# TorchSig Sig53 uses lowercase names
_TORCHSIG_NAMES = [
    "bpsk", "qpsk", "8psk", "16qam", "64qam", "ook", "2fsk", "ofdm-64",
]

_TORCHSIG_TO_CANONICAL = dict(zip(_TORCHSIG_NAMES, CANONICAL_CLASSES))
_CANONICAL_TO_IDX = {name: i for i, name in enumerate(CANONICAL_CLASSES)}


def spectra_waveform_pool():
    """Return SPECTRA waveform instances matching the canonical 8-class set."""
    return [
        sp.BPSK(), sp.QPSK(), sp.PSK8(), sp.QAM16(),
        sp.QAM64(), sp.OOK(), sp.FSK(order=2), sp.OFDM(),
    ]


def torchsig_class_names():
    """Return TorchSig Sig53 class name strings for the canonical 8 classes."""
    return list(_TORCHSIG_NAMES)


def canonical_to_index(name):
    """Map canonical class name to integer index."""
    return _CANONICAL_TO_IDX[name]


def torchsig_to_canonical(torchsig_name):
    """Map TorchSig class string to canonical class name."""
    return _TORCHSIG_TO_CANONICAL[torchsig_name.lower()]
