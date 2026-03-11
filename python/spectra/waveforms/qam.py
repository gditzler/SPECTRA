import numpy as np

from spectra._rust import generate_qam_symbols
from spectra.waveforms.rrc_base import _RRCWaveformBase


class _QAMBase(_RRCWaveformBase):
    """Base class for square QAM waveforms."""

    _order: int = 16

    def _generate_symbols(self, num_symbols, seed):
        return np.array(
            generate_qam_symbols(num_symbols, self._order, seed=seed),
            dtype=np.complex64,
        )


class QAM16(_QAMBase):
    _order = 16

    @property
    def label(self) -> str:
        return "16QAM"


class QAM64(_QAMBase):
    _order = 64

    @property
    def label(self) -> str:
        return "64QAM"


class QAM256(_QAMBase):
    _order = 256

    @property
    def label(self) -> str:
        return "256QAM"


class QAM1024(_QAMBase):
    _order = 1024

    @property
    def label(self) -> str:
        return "1024QAM"


class _CrossQAMBase(_RRCWaveformBase):
    """Base class for non-square cross QAM constellations (32, 128, 512)."""

    _order: int = 32

    @staticmethod
    def _cross_constellation(order: int) -> np.ndarray:
        """Build a cross QAM constellation for non-square orders."""
        side = int(np.ceil(np.sqrt(order)))
        if side % 2 != 0:
            side += 1
        points = []
        for i in range(side):
            for j in range(side):
                re = 2.0 * i - (side - 1)
                im = 2.0 * j - (side - 1)
                points.append(complex(re, im))
        points = np.array(points)
        mags = np.abs(points)
        indices = np.argsort(mags)
        constellation = points[indices[:order]]
        avg_power = np.mean(np.abs(constellation) ** 2)
        if avg_power > 0:
            constellation /= np.sqrt(avg_power)
        return constellation.astype(np.complex64)

    def _generate_symbols(self, num_symbols, seed):
        rng = np.random.default_rng(seed)
        constellation = self._cross_constellation(self._order)
        indices = rng.integers(0, len(constellation), size=num_symbols)
        return constellation[indices]


class QAM32(_CrossQAMBase):
    _order = 32

    @property
    def label(self) -> str:
        return "32QAM"


class QAM128(_CrossQAMBase):
    _order = 128

    @property
    def label(self) -> str:
        return "128QAM"


class QAM512(_CrossQAMBase):
    _order = 512

    @property
    def label(self) -> str:
        return "512QAM"
