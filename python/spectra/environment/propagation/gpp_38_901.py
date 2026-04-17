"""3GPP TR 38.901 path loss models (UMa, UMi, RMa, InH).

Implements path loss + LOS probability + large-scale parameters
(shadow fading, RMS delay spread, Rician K-factor, azimuth arrival
spread ASA) per 3GPP TR 38.901 V17.0.0 §7.4 and §7.5.

Populates `PathLossResult.rms_delay_spread_s`, `k_factor_db` (LOS only),
and `angular_spread_deg` (ASA median) so `link_params_to_impairments()`
can auto-configure downstream fading models.
"""

from __future__ import annotations

import abc
import math

import numpy as np

from spectra.environment.propagation._base import (
    SPEED_OF_LIGHT,
    LOSMode,
    PathLossResult,
    PropagationModel,
    _check_distance_range,
    _check_freq_range,
    _resolve_los,
)


class _GPP38901Base(PropagationModel, abc.ABC):
    """Shared orchestration for 3GPP 38.901 scenarios."""

    # Subclasses override:
    MODEL_NAME: str = "_GPP38901Base"
    FREQ_RANGE_HZ: tuple[float, float] = (500e6, 100e9)
    DISTANCE_RANGE_M: tuple[float, float] = (10.0, 5000.0)

    def __init__(
        self,
        h_bs_m: float,
        h_ut_m: float,
        los_mode: LOSMode = "stochastic",
        strict_range: bool = True,
    ):
        self.h_bs_m = h_bs_m
        self.h_ut_m = h_ut_m
        self.los_mode = los_mode
        self.strict_range = strict_range

    # --- Scenario-specific hooks --------------------------------------

    @abc.abstractmethod
    def _los_probability(self, d_2d_m: float) -> float:
        """LOS probability per TR 38.901 Table 7.4.2-1."""

    @abc.abstractmethod
    def _path_loss_los(
        self, d_3d_m: float, d_2d_m: float, f_ghz: float
    ) -> float:
        """Mean LOS path loss (dB) per TR 38.901 Table 7.4.1-1."""

    @abc.abstractmethod
    def _path_loss_nlos(
        self, d_3d_m: float, d_2d_m: float, f_ghz: float
    ) -> float:
        """Mean NLOS path loss (dB) per TR 38.901 Table 7.4.1-1."""

    @abc.abstractmethod
    def _large_scale_params(
        self, is_los: bool, f_ghz: float
    ) -> tuple[float, float, float, float]:
        """Return (sigma_sf_db, mu_lgDS, sigma_lgDS, asa_deg_median).

        Per TR 38.901 Table 7.5-6. `mu_lgDS` is the lognormal mean
        (log10 of delay spread in seconds); `sigma_lgDS` is its std dev.
        """

    @abc.abstractmethod
    def _k_factor_params(self, f_ghz: float) -> tuple[float, float]:
        """Return (mu_k_db, sigma_k_db) for LOS. Called only when is_los=True."""

    # --- Main entry point --------------------------------------------

    def __call__(
        self, distance_m: float, freq_hz: float, **kwargs
    ) -> PathLossResult:
        if distance_m <= 0:
            raise ValueError("distance_m must be positive")
        _check_freq_range(
            freq_hz, *self.FREQ_RANGE_HZ, self.MODEL_NAME, strict=self.strict_range
        )
        _check_distance_range(
            distance_m,
            *self.DISTANCE_RANGE_M,
            self.MODEL_NAME,
            strict=self.strict_range,
        )

        seed = kwargs.get("seed")
        rng = np.random.default_rng(seed)

        # 2D and 3D distances (h_bs, h_ut are heights above ground)
        d_2d = distance_m
        d_3d = math.sqrt(d_2d ** 2 + (self.h_bs_m - self.h_ut_m) ** 2)
        f_ghz = freq_hz / 1e9

        # LOS resolution
        p_los = self._los_probability(d_2d)
        is_los = _resolve_los(self.los_mode, p_los, rng)

        # Mean path loss
        if is_los:
            pl_mean_db = self._path_loss_los(d_3d, d_2d, f_ghz)
        else:
            pl_mean_db = self._path_loss_nlos(d_3d, d_2d, f_ghz)

        # Large-scale parameters (Table 7.5-6)
        sigma_sf, mu_lgDS, sigma_lgDS, asa_med = self._large_scale_params(
            is_los, f_ghz
        )

        # Shadow fading ~ N(0, sigma_sf)
        sf_db = float(rng.normal(0.0, sigma_sf))

        # Delay spread ~ 10^(mu_lgDS + sigma_lgDS * N(0, 1))
        ds_s = float(10 ** (mu_lgDS + sigma_lgDS * rng.standard_normal()))

        # K-factor (LOS only)
        k_db: float | None = None
        if is_los:
            mu_k, sigma_k = self._k_factor_params(f_ghz)
            k_db = float(rng.normal(mu_k, sigma_k))

        return PathLossResult(
            path_loss_db=pl_mean_db + sf_db,
            shadow_fading_db=sf_db,
            rms_delay_spread_s=ds_s,
            k_factor_db=k_db,
            angular_spread_deg=asa_med,
        )


# ---------------------------------------------------------------------
# UMa — Urban Macro (TR 38.901 Table 7.4.1-1, 7.4.2-1, 7.5-6)
# ---------------------------------------------------------------------


def _c_of_hut_uma(h_ut_m: float) -> float:
    """C(h_UT) factor for UMa LOS probability (Table 7.4.2-1)."""
    if h_ut_m <= 13.0:
        return 0.0
    return ((h_ut_m - 13.0) / 10.0) ** 1.5


class GPP38901UMa(_GPP38901Base):
    """3GPP 38.901 Urban Macro path loss (0.5-100 GHz, 10 m - 5 km)."""

    MODEL_NAME = "GPP38901UMa"
    FREQ_RANGE_HZ = (500e6, 100e9)
    DISTANCE_RANGE_M = (10.0, 5000.0)

    # Effective environment height (TR 38.901 Note 1, Table 7.4.1-1)
    _H_E_M = 1.0

    def _los_probability(self, d_2d_m: float) -> float:
        if d_2d_m <= 18.0:
            return 1.0
        base = (18.0 / d_2d_m) + math.exp(-d_2d_m / 63.0) * (
            1.0 - 18.0 / d_2d_m
        )
        correction = 1.0 + _c_of_hut_uma(self.h_ut_m) * (5.0 / 4.0) * (
            d_2d_m / 100.0
        ) ** 3 * math.exp(-d_2d_m / 150.0)
        return min(1.0, base * correction)

    def _breakpoint_m(self, f_ghz: float) -> float:
        h_bs_prime = self.h_bs_m - self._H_E_M
        h_ut_prime = self.h_ut_m - self._H_E_M
        return 4.0 * h_bs_prime * h_ut_prime * (f_ghz * 1e9) / SPEED_OF_LIGHT

    def _path_loss_los(
        self, d_3d_m: float, d_2d_m: float, f_ghz: float
    ) -> float:
        d_bp = self._breakpoint_m(f_ghz)
        if d_2d_m <= d_bp:
            return 28.0 + 22.0 * math.log10(d_3d_m) + 20.0 * math.log10(f_ghz)
        return (
            28.0
            + 40.0 * math.log10(d_3d_m)
            + 20.0 * math.log10(f_ghz)
            - 9.0 * math.log10(d_bp ** 2 + (self.h_bs_m - self.h_ut_m) ** 2)
        )

    def _path_loss_nlos(
        self, d_3d_m: float, d_2d_m: float, f_ghz: float
    ) -> float:
        pl_nlos_prime = (
            13.54
            + 39.08 * math.log10(d_3d_m)
            + 20.0 * math.log10(f_ghz)
            - 0.6 * (self.h_ut_m - 1.5)
        )
        pl_los = self._path_loss_los(d_3d_m, d_2d_m, f_ghz)
        return max(pl_los, pl_nlos_prime)

    def _large_scale_params(
        self, is_los: bool, f_ghz: float
    ) -> tuple[float, float, float, float]:
        # Table 7.5-6 Part 1 (UMa)
        if is_los:
            mu_lgDS = -6.955 - 0.0963 * math.log10(f_ghz)
            sigma_lgDS = 0.66
            sigma_sf = 4.0
            asa_med = 10.0 ** 1.81  # ~64.6°
        else:
            mu_lgDS = -6.28 - 0.204 * math.log10(f_ghz)
            sigma_lgDS = 0.39
            sigma_sf = 6.0
            asa_med = 10.0 ** 2.08  # ~120°
        return sigma_sf, mu_lgDS, sigma_lgDS, asa_med

    def _k_factor_params(self, f_ghz: float) -> tuple[float, float]:
        # Table 7.5-6 Part 1 (UMa LOS)
        return 9.0, 3.5


# ---------------------------------------------------------------------
# UMi — Urban Micro Street-Canyon (TR 38.901 Table 7.4.1-1, 7.4.2-1, 7.5-6)
# ---------------------------------------------------------------------


class GPP38901UMi(_GPP38901Base):
    """3GPP 38.901 Urban Micro Street-Canyon (0.5-100 GHz, 10 m - 5 km)."""

    MODEL_NAME = "GPP38901UMi"
    FREQ_RANGE_HZ = (500e6, 100e9)
    DISTANCE_RANGE_M = (10.0, 5000.0)

    _H_E_M = 1.0

    def _los_probability(self, d_2d_m: float) -> float:
        # Table 7.4.2-1: no h_UT dependence for UMi
        if d_2d_m <= 18.0:
            return 1.0
        return (18.0 / d_2d_m) + math.exp(-d_2d_m / 36.0) * (1.0 - 18.0 / d_2d_m)

    def _breakpoint_m(self, f_ghz: float) -> float:
        h_bs_prime = self.h_bs_m - self._H_E_M
        h_ut_prime = self.h_ut_m - self._H_E_M
        return 4.0 * h_bs_prime * h_ut_prime * (f_ghz * 1e9) / SPEED_OF_LIGHT

    def _path_loss_los(
        self, d_3d_m: float, d_2d_m: float, f_ghz: float
    ) -> float:
        d_bp = self._breakpoint_m(f_ghz)
        if d_2d_m <= d_bp:
            return 32.4 + 21.0 * math.log10(d_3d_m) + 20.0 * math.log10(f_ghz)
        return (
            32.4
            + 40.0 * math.log10(d_3d_m)
            + 20.0 * math.log10(f_ghz)
            - 9.5 * math.log10(d_bp ** 2 + (self.h_bs_m - self.h_ut_m) ** 2)
        )

    def _path_loss_nlos(
        self, d_3d_m: float, d_2d_m: float, f_ghz: float
    ) -> float:
        pl_nlos_prime = (
            35.3 * math.log10(d_3d_m)
            + 22.4
            + 21.3 * math.log10(f_ghz)
            - 0.3 * (self.h_ut_m - 1.5)
        )
        pl_los = self._path_loss_los(d_3d_m, d_2d_m, f_ghz)
        return max(pl_los, pl_nlos_prime)

    def _large_scale_params(
        self, is_los: bool, f_ghz: float
    ) -> tuple[float, float, float, float]:
        # Table 7.5-6 Part 1 (UMi)
        if is_los:
            mu_lgDS = -0.24 * math.log10(1.0 + f_ghz) - 7.14
            sigma_lgDS = 0.38
            sigma_sf = 4.0
            asa_med = 10.0 ** (-0.08 * math.log10(1.0 + f_ghz) + 1.73)
        else:
            mu_lgDS = -0.24 * math.log10(1.0 + f_ghz) - 6.83
            sigma_lgDS = 0.16 * math.log10(1.0 + f_ghz) + 0.28
            sigma_sf = 7.82
            asa_med = 10.0 ** (-0.08 * math.log10(1.0 + f_ghz) + 1.81)
        return sigma_sf, mu_lgDS, sigma_lgDS, asa_med

    def _k_factor_params(self, f_ghz: float) -> tuple[float, float]:
        # Table 7.5-6 Part 1 (UMi LOS)
        return 9.0, 5.0
