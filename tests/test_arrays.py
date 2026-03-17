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


def test_ula_positions():
    from spectra.arrays.array import ula

    arr = ula(num_elements=4, spacing=0.5, frequency=1e9)
    expected = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]])
    np.testing.assert_allclose(arr.positions, expected)


def test_uca_num_elements():
    from spectra.arrays.array import uca

    arr = uca(num_elements=8, frequency=2.4e9)
    assert arr.num_elements == 8


def test_uca_elements_on_circle():
    from spectra.arrays.array import uca

    arr = uca(num_elements=6, frequency=1e9)
    radii = np.sqrt(arr.positions[:, 0] ** 2 + arr.positions[:, 1] ** 2)
    np.testing.assert_allclose(radii, radii[0], rtol=1e-6)


def test_rectangular_shape():
    from spectra.arrays.array import rectangular

    arr = rectangular(rows=2, cols=3, frequency=1e9)
    assert arr.num_elements == 6


def test_calibration_errors_apply_shape():
    from spectra.arrays.calibration import CalibrationErrors

    cal = CalibrationErrors(
        gain_offsets_db=np.array([0.1, -0.1, 0.05]),
        phase_offsets_rad=np.array([0.01, -0.02, 0.0]),
    )
    sv = np.array([1.0 + 0j, 0.5 + 0.5j, 0.0 + 1.0j])
    sv_cal = cal.apply(sv)
    assert sv_cal.shape == sv.shape
    assert np.issubdtype(sv_cal.dtype, np.complexfloating)


def test_calibration_errors_random():
    from spectra.arrays.calibration import CalibrationErrors

    rng = np.random.default_rng(42)
    cal = CalibrationErrors.random(num_elements=4, gain_std_db=0.5, phase_std_rad=0.05, rng=rng)
    assert cal.gain_offsets_db.shape == (4,)
    assert cal.phase_offsets_rad.shape == (4,)


def test_calibration_errors_zero_is_identity():
    """Zero calibration errors must not change the steering vector."""
    from spectra.arrays.calibration import CalibrationErrors

    cal = CalibrationErrors(
        gain_offsets_db=np.zeros(3),
        phase_offsets_rad=np.zeros(3),
    )
    sv = np.array([1.0 + 0j, 0.5 + 0.5j, 0.0 + 1.0j])
    sv_cal = cal.apply(sv)
    np.testing.assert_allclose(sv_cal, sv, atol=1e-10)


def test_arrays_package_exports():
    import spectra.arrays as arrays

    assert hasattr(arrays, "AntennaArray")
    assert hasattr(arrays, "CalibrationErrors")
    assert hasattr(arrays, "ula")
    assert hasattr(arrays, "uca")
    assert hasattr(arrays, "rectangular")
