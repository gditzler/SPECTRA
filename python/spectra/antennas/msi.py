"""MSI/Planet antenna file parser and element."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from spectra.antennas.base import AntennaElement


class MSIAntennaElement(AntennaElement):
    """Antenna element loaded from an MSI/Planet pattern file.

    The 2D gain surface is built from horizontal and vertical pattern cuts
    using the additive pattern method in dB. Bilinear interpolation via
    scipy.interpolate.RegularGridInterpolator handles arbitrary query angles.

    Args:
        gain_surface_db: 2D array of shape (360, 181) indexed by
            (azimuth_deg, elevation_deg+90). Values in dB relative to peak.
        peak_gain_dbi: Peak antenna gain in dBi (from MSI header).
        frequency_hz: Design frequency in Hz.
        name: Antenna name from MSI header.
    """

    def __init__(
        self,
        gain_surface_db: np.ndarray,
        peak_gain_dbi: float,
        frequency_hz: float,
        name: str = "",
    ):
        self._gain_surface_db = gain_surface_db  # (360, 181)
        self._peak_gain_dbi = peak_gain_dbi
        self._frequency = frequency_hz
        self.name = name
        az_deg = np.arange(360)
        el_idx = np.arange(181)
        self._interp = RegularGridInterpolator(
            (az_deg, el_idx),
            gain_surface_db,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    @property
    def frequency(self) -> float:
        return self._frequency

    @property
    def peak_gain_dbi(self) -> float:
        return self._peak_gain_dbi

    def pattern(self, azimuth: np.ndarray, elevation: np.ndarray) -> np.ndarray:
        azimuth = np.asarray(azimuth, dtype=float)
        elevation = np.asarray(elevation, dtype=float)
        az_b, el_b = np.broadcast_arrays(azimuth, elevation)
        az_deg = np.degrees(az_b) % 360.0
        el_deg_idx = np.degrees(el_b) + 90.0
        points = np.stack([az_deg.ravel(), el_deg_idx.ravel()], axis=-1)
        relative_db = self._interp(points).reshape(az_b.shape)
        total_db = self._peak_gain_dbi + relative_db
        gain_linear = 10.0 ** (total_db / 20.0)
        return gain_linear.astype(complex)


def parse_msi(path: str) -> MSIAntennaElement:
    """Parse an MSI/Planet antenna file and return an MSIAntennaElement.

    Args:
        path: Path to the .msi file.

    Returns:
        MSIAntennaElement with interpolated 2D gain surface.

    Raises:
        ValueError: If the file is missing required sections or has invalid data.
        FileNotFoundError: If the file does not exist.
    """
    with open(path) as f:
        content = f.read()

    lines = content.splitlines()

    name = ""
    frequency_mhz = None
    gain_dbi = 0.0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        upper = stripped.upper()
        if upper.startswith("NAME"):
            name = stripped.split(None, 1)[1] if len(stripped.split()) > 1 else ""
        elif upper.startswith("FREQUENCY"):
            frequency_mhz = float(stripped.split()[1])
        elif upper.startswith("GAIN"):
            gain_dbi = float(stripped.split()[1])

    if frequency_mhz is None:
        raise ValueError("MSI file missing FREQUENCY header field")

    def _parse_section(lines, section_name):
        start = None
        for i, line in enumerate(lines):
            if line.strip().upper() == section_name:
                start = i + 1
                break
        if start is None:
            raise ValueError(f"MSI file missing {section_name} section")
        values = []
        for line in lines[start:]:
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                continue
            if stripped.upper() in ("HORIZONTAL", "VERTICAL"):
                break
            parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                values.append(float(parts[1]))
            except ValueError as e:
                raise ValueError(
                    f"Non-numeric gain in {section_name} section: {line!r}"
                ) from e
        if len(values) != 360:
            raise ValueError(
                f"Expected 360 entries in {section_name} section, got {len(values)}"
            )
        return np.array(values, dtype=float)

    horiz_db = _parse_section(lines, "HORIZONTAL")
    vert_db = _parse_section(lines, "VERTICAL")

    g_peak = max(horiz_db.max(), vert_db.max())

    # Build 181-element vertical cut from the 360-entry MSI vertical array
    # MSI vertical[0] = zenith (el=+90° → el_idx=180), vertical[180] = nadir (el=-90° → el_idx=0)
    vert_cut_181 = np.zeros(181, dtype=float)
    for el_i in range(181):
        v_angle = 180 - el_i
        vert_cut_181[el_i] = vert_db[v_angle % 360]

    # Additive pattern method: G(az, el) = G_h(az) + G_v(el) - G_peak
    gain_surface = (
        horiz_db[:, np.newaxis] + vert_cut_181[np.newaxis, :] - g_peak
    )  # (360, 181)

    frequency_hz = frequency_mhz * 1e6
    return MSIAntennaElement(
        gain_surface_db=gain_surface,
        peak_gain_dbi=gain_dbi,
        frequency_hz=frequency_hz,
        name=name,
    )
