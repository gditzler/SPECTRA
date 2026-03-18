"""Swerling RCS fluctuation models for radar target simulation.

Provides :class:`NonFluctuatingRCS` (Swerling 0/V) and :class:`SwerlingRCS`
(cases I-IV) for generating per-pulse amplitude scale factors.
"""

from __future__ import annotations

import numpy as np


class NonFluctuatingRCS:
    """Non-fluctuating RCS (Swerling case 0/V).

    Returns constant amplitude ``sqrt(sigma)`` for every pulse.

    Args:
        sigma: Mean radar cross-section (linear scale).
    """

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def amplitudes(
        self, num_dwells: int, num_pulses_per_dwell: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Return constant amplitude array, shape ``(num_dwells, num_pulses_per_dwell)``."""
        return np.full((num_dwells, num_pulses_per_dwell), np.sqrt(self.sigma))


class SwerlingRCS:
    """Swerling fluctuating RCS model (cases I-IV).

    Generates amplitude scale factors drawn from the appropriate chi-squared
    distribution:

    - **Cases I, II:** Chi-squared with 2 degrees of freedom (exponential).
    - **Cases III, IV:** Chi-squared with 4 degrees of freedom.
    - **Cases I, III:** Constant within each dwell (scan-to-scan fluctuation).
    - **Cases II, IV:** Independent pulse-to-pulse fluctuation.

    Args:
        case: Swerling case number (1, 2, 3, or 4).
        sigma: Mean radar cross-section (linear scale).
    """

    _VALID_CASES = {1, 2, 3, 4}

    def __init__(self, case: int, sigma: float) -> None:
        if case not in self._VALID_CASES:
            raise ValueError(
                f"Swerling case must be one of {self._VALID_CASES}, got {case}"
            )
        self.case = case
        self.sigma = sigma

    def amplitudes(
        self, num_dwells: int, num_pulses_per_dwell: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Return amplitude scale factors, shape ``(num_dwells, num_pulses_per_dwell)``.

        Cases I/III broadcast a single draw per dwell across all pulses.
        Cases II/IV draw independently for each (dwell, pulse) entry.
        """
        scan_to_scan = self.case in (1, 3)
        dof = 2 if self.case in (1, 2) else 4

        if scan_to_scan:
            rcs_draws = rng.chisquare(dof, size=num_dwells) * (self.sigma / dof)
            rcs_draws = np.repeat(rcs_draws[:, np.newaxis], num_pulses_per_dwell, axis=1)
        else:
            rcs_draws = rng.chisquare(dof, size=(num_dwells, num_pulses_per_dwell)) * (
                self.sigma / dof
            )

        return np.sqrt(rcs_draws)
