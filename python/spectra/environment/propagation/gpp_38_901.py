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
    def _path_loss_los(self, d_3d_m: float, d_2d_m: float, f_ghz: float) -> float:
        """Mean LOS path loss (dB) per TR 38.901 Table 7.4.1-1."""

    @abc.abstractmethod
    def _path_loss_nlos(self, d_3d_m: float, d_2d_m: float, f_ghz: float) -> float:
        """Mean NLOS path loss (dB) per TR 38.901 Table 7.4.1-1."""

    @abc.abstractmethod
    def _large_scale_params(self, is_los: bool, f_ghz: float) -> tuple[float, float, float, float]:
        """Return (sigma_sf_db, mu_lgDS, sigma_lgDS, asa_deg_median).

        Per TR 38.901 Table 7.5-6. `mu_lgDS` is the lognormal mean
        (log10 of delay spread in seconds); `sigma_lgDS` is its std dev.
        """

    @abc.abstractmethod
    def _k_factor_params(self, f_ghz: float) -> tuple[float, float]:
        """Return (mu_k_db, sigma_k_db) for LOS. Called only when is_los=True."""

    # --- Main entry point --------------------------------------------

    def __call__(self, distance_m: float, freq_hz: float, **kwargs) -> PathLossResult:
        if distance_m <= 0:
            raise ValueError("distance_m must be positive")
        _check_freq_range(freq_hz, *self.FREQ_RANGE_HZ, self.MODEL_NAME, strict=self.strict_range)
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
        d_3d = math.sqrt(d_2d**2 + (self.h_bs_m - self.h_ut_m) ** 2)
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
        sigma_sf, mu_lgDS, sigma_lgDS, asa_med = self._large_scale_params(is_los, f_ghz)

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
        base = (18.0 / d_2d_m) + math.exp(-d_2d_m / 63.0) * (1.0 - 18.0 / d_2d_m)
        correction = 1.0 + _c_of_hut_uma(self.h_ut_m) * (5.0 / 4.0) * (
            d_2d_m / 100.0
        ) ** 3 * math.exp(-d_2d_m / 150.0)
        return min(1.0, base * correction)

    def _breakpoint_m(self, f_ghz: float) -> float:
        h_bs_prime = self.h_bs_m - self._H_E_M
        h_ut_prime = self.h_ut_m - self._H_E_M
        return 4.0 * h_bs_prime * h_ut_prime * (f_ghz * 1e9) / SPEED_OF_LIGHT

    def _path_loss_los(self, d_3d_m: float, d_2d_m: float, f_ghz: float) -> float:
        d_bp = self._breakpoint_m(f_ghz)
        if d_2d_m <= d_bp:
            return 28.0 + 22.0 * math.log10(d_3d_m) + 20.0 * math.log10(f_ghz)
        return (
            28.0
            + 40.0 * math.log10(d_3d_m)
            + 20.0 * math.log10(f_ghz)
            - 9.0 * math.log10(d_bp**2 + (self.h_bs_m - self.h_ut_m) ** 2)
        )

    def _path_loss_nlos(self, d_3d_m: float, d_2d_m: float, f_ghz: float) -> float:
        pl_nlos_prime = (
            13.54
            + 39.08 * math.log10(d_3d_m)
            + 20.0 * math.log10(f_ghz)
            - 0.6 * (self.h_ut_m - 1.5)
        )
        pl_los = self._path_loss_los(d_3d_m, d_2d_m, f_ghz)
        return max(pl_los, pl_nlos_prime)

    def _large_scale_params(self, is_los: bool, f_ghz: float) -> tuple[float, float, float, float]:
        # Table 7.5-6 Part 1 (UMa)
        if is_los:
            mu_lgDS = -6.955 - 0.0963 * math.log10(f_ghz)
            sigma_lgDS = 0.66
            sigma_sf = 4.0
            asa_med = 10.0**1.81  # ~64.6°
        else:
            mu_lgDS = -6.28 - 0.204 * math.log10(f_ghz)
            sigma_lgDS = 0.39
            sigma_sf = 6.0
            asa_med = 10.0**2.08  # ~120°
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

    def _path_loss_los(self, d_3d_m: float, d_2d_m: float, f_ghz: float) -> float:
        d_bp = self._breakpoint_m(f_ghz)
        if d_2d_m <= d_bp:
            return 32.4 + 21.0 * math.log10(d_3d_m) + 20.0 * math.log10(f_ghz)
        return (
            32.4
            + 40.0 * math.log10(d_3d_m)
            + 20.0 * math.log10(f_ghz)
            - 9.5 * math.log10(d_bp**2 + (self.h_bs_m - self.h_ut_m) ** 2)
        )

    def _path_loss_nlos(self, d_3d_m: float, d_2d_m: float, f_ghz: float) -> float:
        pl_nlos_prime = (
            35.3 * math.log10(d_3d_m) + 22.4 + 21.3 * math.log10(f_ghz) - 0.3 * (self.h_ut_m - 1.5)
        )
        pl_los = self._path_loss_los(d_3d_m, d_2d_m, f_ghz)
        return max(pl_los, pl_nlos_prime)

    def _large_scale_params(self, is_los: bool, f_ghz: float) -> tuple[float, float, float, float]:
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


# ---------------------------------------------------------------------
# RMa — Rural Macro (TR 38.901 Table 7.4.1-1, 7.4.2-1, 7.5-6)
# ---------------------------------------------------------------------


class GPP38901RMa(_GPP38901Base):
    """3GPP 38.901 Rural Macro (0.5-30 GHz, 10 m - 10 km).

    Extra parameters:
        h_building_m: average building height (default 5 m, per Note 3).
        w_street_m: average street width (default 20 m, per Note 3).
    """

    MODEL_NAME = "GPP38901RMa"
    FREQ_RANGE_HZ = (500e6, 30e9)
    DISTANCE_RANGE_M = (10.0, 10000.0)

    def __init__(
        self,
        h_bs_m: float,
        h_ut_m: float,
        h_building_m: float = 5.0,
        w_street_m: float = 20.0,
        los_mode: LOSMode = "stochastic",
        strict_range: bool = True,
    ):
        super().__init__(
            h_bs_m=h_bs_m,
            h_ut_m=h_ut_m,
            los_mode=los_mode,
            strict_range=strict_range,
        )
        self.h_building_m = h_building_m
        self.w_street_m = w_street_m

    def _los_probability(self, d_2d_m: float) -> float:
        # Table 7.4.2-1: RMa
        if d_2d_m <= 10.0:
            return 1.0
        return math.exp(-(d_2d_m - 10.0) / 1000.0)

    def _breakpoint_m(self, f_ghz: float) -> float:
        # d_BP = 2*pi*h_BS*h_UT*f_c/c   (note: 2*pi, not 4)
        return 2.0 * math.pi * self.h_bs_m * self.h_ut_m * (f_ghz * 1e9) / SPEED_OF_LIGHT

    def _path_loss_los(self, d_3d_m: float, d_2d_m: float, f_ghz: float) -> float:
        h_b = self.h_building_m
        pl_1 = (
            20.0 * math.log10(40.0 * math.pi * d_3d_m * f_ghz / 3.0)
            + min(0.03 * h_b**1.72, 10.0) * math.log10(d_3d_m)
            - min(0.044 * h_b**1.72, 14.77)
            + 0.002 * math.log10(h_b) * d_3d_m
        )
        d_bp = self._breakpoint_m(f_ghz)
        if d_2d_m <= d_bp:
            return pl_1
        pl_at_bp = (
            20.0 * math.log10(40.0 * math.pi * d_bp * f_ghz / 3.0)
            + min(0.03 * h_b**1.72, 10.0) * math.log10(d_bp)
            - min(0.044 * h_b**1.72, 14.77)
            + 0.002 * math.log10(h_b) * d_bp
        )
        return pl_at_bp + 40.0 * math.log10(d_3d_m / d_bp)

    def _path_loss_nlos(self, d_3d_m: float, d_2d_m: float, f_ghz: float) -> float:
        h_b = self.h_building_m
        w = self.w_street_m
        pl_nlos_prime = (
            161.04
            - 7.1 * math.log10(w)
            + 7.5 * math.log10(h_b)
            - (24.37 - 3.7 * (h_b / self.h_bs_m) ** 2) * math.log10(self.h_bs_m)
            + (43.42 - 3.1 * math.log10(self.h_bs_m)) * (math.log10(d_3d_m) - 3.0)
            + 20.0 * math.log10(f_ghz)
            - (3.2 * (math.log10(11.75 * self.h_ut_m)) ** 2 - 4.97)
        )
        pl_los = self._path_loss_los(d_3d_m, d_2d_m, f_ghz)
        return max(pl_los, pl_nlos_prime)

    def _large_scale_params(self, is_los: bool, f_ghz: float) -> tuple[float, float, float, float]:
        # Table 7.5-6 Part 2 (RMa) — RMa values are frequency-independent
        if is_los:
            return 4.0, -7.49, 0.55, 10.0**1.52
        return 8.0, -7.43, 0.48, 10.0**1.52

    def _k_factor_params(self, f_ghz: float) -> tuple[float, float]:
        # Table 7.5-6 Part 2 (RMa LOS)
        return 7.0, 4.0


# ---------------------------------------------------------------------
# InH — Indoor Hotspot (TR 38.901 Table 7.4.1-1, 7.4.2-1, 7.5-6)
# ---------------------------------------------------------------------


class GPP38901InH(_GPP38901Base):
    """3GPP 38.901 Indoor Hotspot (0.5-100 GHz, 1 m - 150 m).

    Supports two variants:
        variant="mixed_office"  — mixed office environment (default)
        variant="open_office"   — open office environment
    """

    MODEL_NAME = "GPP38901InH"
    FREQ_RANGE_HZ = (500e6, 100e9)
    DISTANCE_RANGE_M = (1.0, 150.0)

    _VALID_VARIANTS = {"mixed_office", "open_office"}

    def __init__(
        self,
        h_bs_m: float = 3.0,
        h_ut_m: float = 1.0,
        variant: str = "mixed_office",
        los_mode: LOSMode = "stochastic",
        strict_range: bool = True,
    ):
        if variant not in self._VALID_VARIANTS:
            raise ValueError(f"variant must be one of {self._VALID_VARIANTS}, got '{variant}'")
        super().__init__(
            h_bs_m=h_bs_m,
            h_ut_m=h_ut_m,
            los_mode=los_mode,
            strict_range=strict_range,
        )
        self.variant = variant

    def _los_probability(self, d_2d_m: float) -> float:
        # Table 7.4.2-1 (InH)
        if self.variant == "mixed_office":
            if d_2d_m <= 1.2:
                return 1.0
            if d_2d_m <= 6.5:
                return math.exp(-(d_2d_m - 1.2) / 4.7)
            return math.exp(-(d_2d_m - 6.5) / 32.6) * 0.32
        # open_office
        if d_2d_m <= 5.0:
            return 1.0
        if d_2d_m <= 49.0:
            return math.exp(-(d_2d_m - 5.0) / 70.8)
        return math.exp(-(d_2d_m - 49.0) / 211.7) * 0.54

    def _path_loss_los(self, d_3d_m: float, d_2d_m: float, f_ghz: float) -> float:
        # Table 7.4.1-1 InH LOS (single-slope)
        return 32.4 + 17.3 * math.log10(d_3d_m) + 20.0 * math.log10(f_ghz)

    def _path_loss_nlos(self, d_3d_m: float, d_2d_m: float, f_ghz: float) -> float:
        pl_nlos = 38.3 * math.log10(d_3d_m) + 17.30 + 24.9 * math.log10(f_ghz)
        pl_los = self._path_loss_los(d_3d_m, d_2d_m, f_ghz)
        return max(pl_los, pl_nlos)

    def _large_scale_params(self, is_los: bool, f_ghz: float) -> tuple[float, float, float, float]:
        # Table 7.5-6 Part 2 (Indoor Office)
        if is_los:
            mu_lgDS = -0.01 * math.log10(1.0 + f_ghz) - 7.692
            sigma_lgDS = 0.18
            sigma_sf = 3.0
            asa_med = 10.0 ** (-0.19 * math.log10(1.0 + f_ghz) + 1.781)
        else:
            mu_lgDS = -0.28 * math.log10(1.0 + f_ghz) - 7.173
            sigma_lgDS = 0.10 * math.log10(1.0 + f_ghz) + 0.055
            sigma_sf = 8.03
            asa_med = 10.0 ** (-0.11 * math.log10(1.0 + f_ghz) + 1.863)
        return sigma_sf, mu_lgDS, sigma_lgDS, asa_med

    def _k_factor_params(self, f_ghz: float) -> tuple[float, float]:
        # Table 7.5-6 Part 2 (InH LOS)
        return 7.0, 4.0
