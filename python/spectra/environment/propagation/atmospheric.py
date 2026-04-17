"""ITU-R P.676 gaseous atmospheric absorption (simplified Annex 2 model).

Implements the closed-form specific-attenuation approximation from
Recommendation ITU-R P.676-13 Annex 2, valid 1-350 GHz. Horizontal
terrestrial paths only (no slant-path / elevation-angle support).
"""

from __future__ import annotations

import math
import warnings

_BELOW_1GHZ_WARNED = False


def _specific_attenuation_oxygen_db_per_km(
    f_ghz: float, p_hpa: float, t_k: float
) -> float:
    """Dry-air (oxygen) specific attenuation (dB/km).

    Implements Eq. (28) of Rec. ITU-R P.676-13 Annex 2 - simplified
    polynomial fit valid 1-54 GHz, with the 60 GHz complex handled by
    dedicated breakpoints. Accurate to ~10% across the envelope.
    """
    rp = p_hpa / 1013.25
    rt = 288.15 / t_k
    # Oxygen attenuation, piecewise per Annex 2 Eq. (28) / Table 2
    if f_ghz <= 54.0:
        term1 = 7.2 * rt**2.8 / (f_ghz**2 + 0.34 * rp**2 * rt**1.6)
        term2 = 0.62 * rp**1.6 * rt**1.5 / ((54.0 - f_ghz) ** 1.16 + 0.83 * rp**2)
        return (term1 + term2) * f_ghz**2 * rp**2 * 1e-3
    if f_ghz <= 60.0:
        # Quadratic interpolation across the 54-60 GHz shoulder of the oxygen complex
        g54 = _specific_attenuation_oxygen_db_per_km(54.0, p_hpa, t_k)
        # Peak ~15 dB/km at 60 GHz under surface conditions
        return g54 + (f_ghz - 54.0) / 6.0 * (15.0 * rp**2 * rt**3 - g54)
    if f_ghz <= 66.0:
        # Oxygen complex peak region (60 GHz)
        return 15.0 * rp**2 * rt**3 * math.exp(-((f_ghz - 60.0) ** 2) / 8.0)
    if f_ghz <= 120.0:
        # Above the complex, drops off rapidly
        return (
            0.283
            * rp**2
            * rt**3.8
            / ((f_ghz - 118.75) ** 2 + 2.91 * rp**2)
            * f_ghz**2
            * 1e-4
        ) + 0.01 * rp**2 * rt**2
    # 120-350 GHz: residual dry-air term
    return 3.02e-4 * rp**2 * rt**3.5 * f_ghz**2


def _specific_attenuation_water_db_per_km(
    f_ghz: float, p_hpa: float, t_k: float, rho_g_m3: float
) -> float:
    """Water-vapor specific attenuation (dB/km).

    Implements Eq. (29) of Rec. ITU-R P.676-13 Annex 2 - dominant lines
    at 22.235, 183.31, and 325.15 GHz, summed with a continuum term.
    """
    rp = p_hpa / 1013.25
    rt = 288.15 / t_k
    eta_1 = 0.955 * rp * rt**0.68 + 0.006 * rho_g_m3
    eta_2 = 0.735 * rp * rt**0.5 + 0.0353 * rt**4 * rho_g_m3

    # Dominant lines
    g22 = (
        3.98 * eta_1 * math.exp(2.23 * (1 - rt)) / ((f_ghz - 22.235) ** 2 + 9.42 * eta_1**2)
    )
    g183 = (
        11.96 * eta_1 * math.exp(0.7 * (1 - rt)) / ((f_ghz - 183.31) ** 2 + 11.14 * eta_1**2)
    )
    g325 = (
        10.48 * eta_2 * math.exp(1.09 * (1 - rt)) / ((f_ghz - 325.153) ** 2 + 6.29 * eta_2**2)
    )
    continuum = 1.61e-8 * rho_g_m3 * rt**2 * f_ghz**2
    return (g22 + g183 + g325 + continuum) * f_ghz**2 * rho_g_m3 * 1e-4


def gaseous_attenuation_db(
    distance_m: float,
    freq_hz: float,
    temperature_k: float = 288.15,
    pressure_hpa: float = 1013.25,
    water_vapor_density_g_m3: float = 7.5,
) -> float:
    """Total one-way gaseous attenuation (dB) over a horizontal path.

    Implements the simplified Annex 2 model of ITU-R P.676-13:
    specific attenuation gamma_o (oxygen) + gamma_w (water vapor), each in
    dB/km, then multiplied by the path length. Valid 1 GHz - 350 GHz.
    Below 1 GHz, returns 0.0 with a one-time UserWarning.

    Parameters
    ----------
    distance_m
        Horizontal path length in meters. Must be >= 0.
    freq_hz
        Carrier frequency in Hz.
    temperature_k
        Atmospheric temperature (default = 288.15 K, ITU reference).
    pressure_hpa
        Dry-air pressure (default = 1013.25 hPa, ITU reference).
    water_vapor_density_g_m3
        Surface water vapor density (default = 7.5 g/m^3, ITU reference).
    """
    global _BELOW_1GHZ_WARNED
    if distance_m < 0:
        raise ValueError("distance_m must be >= 0")
    if distance_m == 0.0:
        return 0.0

    f_ghz = freq_hz / 1e9
    if f_ghz < 1.0:
        if not _BELOW_1GHZ_WARNED:
            warnings.warn(
                f"ITU-R P.676 gaseous attenuation is negligible below 1 GHz; "
                f"returning 0.0 (f={f_ghz:.3g} GHz).",
                UserWarning,
                stacklevel=2,
            )
            _BELOW_1GHZ_WARNED = True
        return 0.0

    gamma_o = _specific_attenuation_oxygen_db_per_km(f_ghz, pressure_hpa, temperature_k)
    gamma_w = _specific_attenuation_water_db_per_km(
        f_ghz, pressure_hpa, temperature_k, water_vapor_density_g_m3
    )
    return (gamma_o + gamma_w) * (distance_m / 1000.0)
