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


def test_isotropic_element_unity_gain():
    from spectra.antennas.isotropic import IsotropicElement

    elem = IsotropicElement(frequency=2.4e9)
    az = np.linspace(0, 2 * np.pi, 36)
    el = np.linspace(-np.pi / 2, np.pi / 2, 36)
    gain = elem.pattern(az, el)
    assert gain.shape == (36,)
    np.testing.assert_array_equal(gain, 1.0 + 0j)


def test_isotropic_frequency_property():
    from spectra.antennas.isotropic import IsotropicElement

    elem = IsotropicElement(frequency=900e6)
    assert elem.frequency == 900e6
