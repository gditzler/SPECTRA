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


def test_short_dipole_z_axis_broadside():
    """Gain is maximum (=1) at elevation=0 (equatorial plane), zero at poles."""
    from spectra.antennas.dipole import ShortDipoleElement

    elem = ShortDipoleElement(axis="z", frequency=300e6)
    az = np.array([0.0])
    el = np.array([0.0])
    gain = elem.pattern(az, el)
    np.testing.assert_allclose(np.abs(gain), [1.0], atol=1e-6)


def test_short_dipole_z_axis_pole_null():
    """Gain is zero at elevation=pi/2 (pole, along dipole axis)."""
    from spectra.antennas.dipole import ShortDipoleElement

    elem = ShortDipoleElement(axis="z", frequency=300e6)
    az = np.array([0.0])
    el = np.array([np.pi / 2])
    gain = elem.pattern(az, el)
    np.testing.assert_allclose(np.abs(gain), [0.0], atol=1e-6)


def test_half_wave_dipole_z_broadside():
    """Half-wave dipole: gain at broadside should be ~1 (normalized)."""
    from spectra.antennas.dipole import HalfWaveDipoleElement

    elem = HalfWaveDipoleElement(axis="z", frequency=300e6)
    az = np.array([0.0])
    el = np.array([0.0])
    gain = elem.pattern(az, el)
    assert np.abs(gain[0]) > 0.9


def test_dipole_returns_complex():
    from spectra.antennas.dipole import ShortDipoleElement

    elem = ShortDipoleElement(axis="z", frequency=300e6)
    gain = elem.pattern(np.array([0.0]), np.array([0.0]))
    assert np.issubdtype(gain.dtype, np.complexfloating)


def test_cosine_power_boresight_max():
    """Gain at boresight (elevation=pi/2) should equal the linear peak gain."""
    from spectra.antennas.cosine_power import CosinePowerElement

    elem = CosinePowerElement(exponent=1.5, peak_gain_dbi=3.0, frequency=2.4e9)
    gain = elem.pattern(np.array([0.0]), np.array([np.pi / 2]))
    peak_linear = 10 ** (3.0 / 10.0)
    np.testing.assert_allclose(np.abs(gain[0]), peak_linear, rtol=1e-5)


def test_cosine_power_back_hemisphere_zero():
    """Gain is zero for theta_off > pi/2 (back hemisphere)."""
    from spectra.antennas.cosine_power import CosinePowerElement

    elem = CosinePowerElement(exponent=1.5, peak_gain_dbi=0.0, frequency=2.4e9)
    gain = elem.pattern(np.array([0.0]), np.array([-np.pi / 2]))
    np.testing.assert_allclose(np.abs(gain[0]), 0.0, atol=1e-6)


def test_cosine_power_returns_complex():
    from spectra.antennas.cosine_power import CosinePowerElement

    elem = CosinePowerElement(exponent=2.0, frequency=1e9)
    gain = elem.pattern(np.array([0.0, 1.0]), np.array([0.0, 0.5]))
    assert np.issubdtype(gain.dtype, np.complexfloating)


import os


def _write_minimal_msi(path: str, freq_mhz: float = 2400.0, gain_dbi: float = 3.0) -> None:
    """Write a valid minimal MSI file with flat horizontal and vertical patterns."""
    lines = [
        "NAME          TestAntenna",
        f"FREQUENCY     {freq_mhz:.0f}",
        f"GAIN          {gain_dbi:.1f}",
        "TILT          0",
        "POLARIZATION  V",
        "",
        "HORIZONTAL",
    ]
    for angle in range(360):
        lines.append(f"{angle}   0.0")
    lines.append("")
    lines.append("VERTICAL")
    for angle in range(360):
        lines.append(f"{angle}   0.0")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def test_parse_msi_returns_msi_element(tmp_path):
    from spectra.antennas.msi import MSIAntennaElement, parse_msi

    msi_file = str(tmp_path / "test.msi")
    _write_minimal_msi(msi_file, freq_mhz=2400.0, gain_dbi=3.0)
    elem = parse_msi(msi_file)
    assert isinstance(elem, MSIAntennaElement)


def test_msi_frequency_in_hz(tmp_path):
    from spectra.antennas.msi import parse_msi

    msi_file = str(tmp_path / "test.msi")
    _write_minimal_msi(msi_file, freq_mhz=900.0, gain_dbi=6.0)
    elem = parse_msi(msi_file)
    assert elem.frequency == pytest.approx(900e6)


def test_msi_peak_gain_dbi(tmp_path):
    from spectra.antennas.msi import parse_msi

    msi_file = str(tmp_path / "test.msi")
    _write_minimal_msi(msi_file, freq_mhz=2400.0, gain_dbi=5.5)
    elem = parse_msi(msi_file)
    assert elem.peak_gain_dbi == pytest.approx(5.5)


def test_msi_pattern_returns_complex(tmp_path):
    from spectra.antennas.msi import parse_msi

    msi_file = str(tmp_path / "test.msi")
    _write_minimal_msi(msi_file, freq_mhz=2400.0, gain_dbi=3.0)
    elem = parse_msi(msi_file)
    az = np.array([0.0, np.pi / 2, np.pi])
    el = np.array([0.0, 0.0, 0.0])
    gain = elem.pattern(az, el)
    assert gain.shape == (3,)
    assert np.issubdtype(gain.dtype, np.complexfloating)


def test_msi_flat_pattern_uniform_gain(tmp_path):
    """Flat 0 dB relative pattern → all gains equal peak_gain_linear."""
    from spectra.antennas.msi import parse_msi

    msi_file = str(tmp_path / "test.msi")
    _write_minimal_msi(msi_file, freq_mhz=2400.0, gain_dbi=3.0)
    elem = parse_msi(msi_file)
    az = np.linspace(0, 2 * np.pi, 10)
    el = np.zeros(10)
    gain = elem.pattern(az, el)
    peak_linear = 10 ** (3.0 / 10.0)
    np.testing.assert_allclose(np.abs(gain), peak_linear, rtol=1e-4)


def test_msi_missing_section_raises(tmp_path):
    from spectra.antennas.msi import parse_msi

    bad_file = str(tmp_path / "bad.msi")
    with open(bad_file, "w") as f:
        f.write("NAME   Test\nFREQUENCY  900\nGAIN  0\n\nHORIZONTAL\n")
        for i in range(360):
            f.write(f"{i}  0.0\n")
        # Missing VERTICAL section

    with pytest.raises(ValueError, match="VERTICAL"):
        parse_msi(bad_file)


def test_antennas_package_exports():
    import spectra.antennas as antennas

    assert hasattr(antennas, "AntennaElement")
    assert hasattr(antennas, "IsotropicElement")
    assert hasattr(antennas, "ShortDipoleElement")
    assert hasattr(antennas, "HalfWaveDipoleElement")
    assert hasattr(antennas, "CosinePowerElement")
    assert hasattr(antennas, "MSIAntennaElement")
    assert hasattr(antennas, "parse_msi")
