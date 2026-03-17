import numpy as np
import pytest


def test_cannot_instantiate_antenna_element_abc():
    from spectra.antennas.base import AntennaElement

    class Incomplete(AntennaElement):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_concrete_antenna_element_interface():
    from spectra.antennas.base import AntennaElement

    class Minimal(AntennaElement):
        @property
        def frequency(self):
            return 2.4e9

        def pattern(self, azimuth, elevation):
            return np.ones_like(azimuth, dtype=complex)

    elem = Minimal()
    az = np.array([0.0, np.pi / 4])
    el = np.array([0.0, 0.0])
    gain = elem.pattern(az, el)
    assert gain.shape == (2,)
    assert gain.dtype == complex or np.issubdtype(gain.dtype, np.complexfloating)
