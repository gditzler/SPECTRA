import numpy as np

from spectra._rust import (
    generate_8psk_symbols,
    generate_bpsk_symbols,
    generate_psk_symbols,
    generate_qpsk_symbols,
)
from spectra.waveforms.rrc_base import _RRCWaveformBase


class QPSK(_RRCWaveformBase):
    @property
    def label(self) -> str:
        return "QPSK"

    def _generate_symbols(self, num_symbols, seed):
        return np.array(generate_qpsk_symbols(num_symbols, seed=seed), dtype=np.complex64)


class BPSK(_RRCWaveformBase):
    @property
    def label(self) -> str:
        return "BPSK"

    def _generate_symbols(self, num_symbols, seed):
        return np.array(generate_bpsk_symbols(num_symbols, seed=seed), dtype=np.complex64)


class PSK8(_RRCWaveformBase):
    @property
    def label(self) -> str:
        return "8PSK"

    def _generate_symbols(self, num_symbols, seed):
        return np.array(generate_8psk_symbols(num_symbols, seed=seed), dtype=np.complex64)


class _PSKBase(_RRCWaveformBase):
    """Base class for higher-order M-PSK waveforms using generic generator."""

    _order: int = 16

    def _generate_symbols(self, num_symbols, seed):
        return np.array(
            generate_psk_symbols(num_symbols, self._order, seed=seed),
            dtype=np.complex64,
        )


class PSK16(_PSKBase):
    _order = 16

    @property
    def label(self) -> str:
        return "16PSK"


class PSK32(_PSKBase):
    _order = 32

    @property
    def label(self) -> str:
        return "32PSK"


class PSK64(_PSKBase):
    _order = 64

    @property
    def label(self) -> str:
        return "64PSK"
