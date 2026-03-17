import numpy as np
import pytest


def test_antenna_array_num_elements():
    from spectra.antennas.isotropic import IsotropicElement
    from spectra.arrays.array import AntennaArray

    positions = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    elem = IsotropicElement(frequency=1e9)
    arr = AntennaArray(positions=positions, elements=elem, reference_frequency=1e9)
    assert arr.num_elements == 3


def test_steering_vector_single_angle_shape():
    from spectra.antennas.isotropic import IsotropicElement
    from spectra.arrays.array import AntennaArray

    positions = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    elem = IsotropicElement(frequency=1e9)
    arr = AntennaArray(positions=positions, elements=elem, reference_frequency=1e9)
    sv = arr.steering_vector(azimuth=0.0, elevation=0.0)
    assert sv.shape == (3,)
    assert np.issubdtype(sv.dtype, np.complexfloating)


def test_steering_vector_multiple_angles_shape():
    from spectra.antennas.isotropic import IsotropicElement
    from spectra.arrays.array import AntennaArray

    positions = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    elem = IsotropicElement(frequency=1e9)
    arr = AntennaArray(positions=positions, elements=elem, reference_frequency=1e9)
    azimuths = np.array([0.0, np.pi / 4, np.pi / 2])
    elevations = np.zeros(3)
    sv = arr.steering_vector(azimuth=azimuths, elevation=elevations)
    assert sv.shape == (3, 3)


def test_steering_vector_isotropic_ula_phase():
    """For an isotropic ULA along x-axis, the inter-element phase shift at
    broadside (az=0, el=0) should be 2*pi*spacing (spacing in wavelengths)."""
    from spectra.antennas.isotropic import IsotropicElement
    from spectra.arrays.array import AntennaArray

    spacing = 0.5  # half-wavelength
    positions = np.array([[0.0, 0.0], [spacing, 0.0], [2 * spacing, 0.0]])
    elem = IsotropicElement(frequency=1e9)
    arr = AntennaArray(positions=positions, elements=elem, reference_frequency=1e9)
    sv = arr.steering_vector(azimuth=0.0, elevation=0.0)
    # Phase difference between adjacent elements should be 2*pi*spacing
    phase_diff = np.angle(sv[1]) - np.angle(sv[0])
    expected = 2 * np.pi * spacing
    np.testing.assert_allclose(phase_diff % (2 * np.pi), expected % (2 * np.pi), atol=1e-5)


def test_antenna_array_per_element_list():
    """When elements is a list, each element gets its own pattern applied."""
    from spectra.antennas.isotropic import IsotropicElement
    from spectra.arrays.array import AntennaArray

    positions = np.array([[0.0, 0.0], [0.5, 0.0]])
    elements = [IsotropicElement(frequency=1e9), IsotropicElement(frequency=1e9)]
    arr = AntennaArray(positions=positions, elements=elements, reference_frequency=1e9)
    sv = arr.steering_vector(azimuth=0.0, elevation=0.0)
    assert sv.shape == (2,)
