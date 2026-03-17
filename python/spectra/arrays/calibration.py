# python/spectra/arrays/calibration.py
"""Per-element calibration error model for antenna arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CalibrationErrors:
    """Per-element gain and phase calibration offsets.

    Args:
        gain_offsets_db: Per-element gain offsets in dB, shape (N,).
        phase_offsets_rad: Per-element phase offsets in radians, shape (N,).

    Example::

        cal = CalibrationErrors.random(num_elements=8, gain_std_db=0.5)
        sv_cal = cal.apply(steering_vector)
    """

    gain_offsets_db: np.ndarray   # (N,)
    phase_offsets_rad: np.ndarray  # (N,)

    def __post_init__(self) -> None:
        if self.gain_offsets_db.ndim != 1 or self.phase_offsets_rad.ndim != 1:
            raise ValueError(
                "gain_offsets_db and phase_offsets_rad must be 1-D arrays"
            )
        if len(self.gain_offsets_db) != len(self.phase_offsets_rad):
            raise ValueError(
                f"gain_offsets_db length {len(self.gain_offsets_db)} != "
                f"phase_offsets_rad length {len(self.phase_offsets_rad)}"
            )

    @classmethod
    def random(
        cls,
        num_elements: int,
        gain_std_db: float = 0.5,
        phase_std_rad: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ) -> "CalibrationErrors":
        """Generate random calibration errors from zero-mean Gaussians.

        Args:
            num_elements: Number of array elements.
            gain_std_db: Standard deviation of gain offsets in dB.
            phase_std_rad: Standard deviation of phase offsets in radians.
            rng: NumPy random generator. If None, uses default_rng().

        Returns:
            CalibrationErrors with random offsets.
        """
        if rng is None:
            rng = np.random.default_rng()
        gain_offsets_db = rng.normal(0.0, gain_std_db, size=num_elements)
        phase_offsets_rad = rng.normal(0.0, phase_std_rad, size=num_elements)
        return cls(
            gain_offsets_db=gain_offsets_db,
            phase_offsets_rad=phase_offsets_rad,
        )

    def apply(self, steering_vector: np.ndarray) -> np.ndarray:
        """Apply calibration errors to a steering vector.

        Implements a_cal = diag(gain_linear * exp(j*phase)) @ a.

        Args:
            steering_vector: Steering vector of shape (N,) or (N, M).

        Returns:
            Calibrated steering vector with the same shape.
        """
        # Gain offsets in dB are field-amplitude offsets → divide by 20
        gain_linear = 10.0 ** (self.gain_offsets_db / 20.0)
        diag = gain_linear * np.exp(1j * self.phase_offsets_rad)  # (N,)
        if steering_vector.ndim == 1:
            return diag * steering_vector
        return diag[:, np.newaxis] * steering_vector
