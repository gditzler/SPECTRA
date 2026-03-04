import numpy as np

from spectra._rust import generate_ask_symbols
from spectra.waveforms.rrc_base import _RRCWaveformBase


class _ASKBase(_RRCWaveformBase):
    """Base class for M-ary ASK waveforms."""

    _order: int = 2

    def _generate_symbols(self, num_symbols, seed):
        return np.array(
            generate_ask_symbols(num_symbols, self._order, seed=seed),
            dtype=np.complex64,
        )


class OOK(_ASKBase):
    _order = 2

    @property
    def label(self) -> str:
        return "OOK"


class ASK4(_ASKBase):
    _order = 4

    @property
    def label(self) -> str:
        return "4ASK"


class ASK8(_ASKBase):
    _order = 8

    @property
    def label(self) -> str:
        return "8ASK"


class ASK16(_ASKBase):
    _order = 16

    @property
    def label(self) -> str:
        return "16ASK"


class ASK32(_ASKBase):
    _order = 32

    @property
    def label(self) -> str:
        return "32ASK"


class ASK64(_ASKBase):
    _order = 64

    @property
    def label(self) -> str:
        return "64ASK"
