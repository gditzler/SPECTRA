"""Generic linear Kalman filter and radar convenience factories.

:class:`KalmanFilter` is state-dimension agnostic — it works for any
``(n, m)`` state/measurement pair. :func:`ConstantVelocityKF` returns a
pre-configured instance for range-only radar tracking.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class KalmanFilter:
    """Generic discrete-time linear Kalman filter.

    Args:
        F: State transition matrix, shape ``(n, n)``.
        H: Measurement matrix, shape ``(m, n)``.
        Q: Process noise covariance, shape ``(n, n)``.
        R: Measurement noise covariance, shape ``(m, m)``.
        x0: Initial state, shape ``(n,)``. Defaults to zeros.
        P0: Initial covariance, shape ``(n, n)``. Defaults to identity.
    """

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
    ) -> None:
        self._F = np.asarray(F, dtype=float)
        self._H = np.asarray(H, dtype=float)
        self._Q = np.asarray(Q, dtype=float)
        self._R = np.asarray(R, dtype=float)

        n = self._F.shape[0]
        self._x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()
        self._P = np.eye(n) if P0 is None else np.asarray(P0, dtype=float).copy()

    @property
    def state(self) -> np.ndarray:
        return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._P.copy()

    def predict(self) -> np.ndarray:
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        return self._x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        y = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        n = len(self._x)
        self._P = (np.eye(n) - K @ self._H) @ self._P
        return self._x.copy()

    def step(self, z: np.ndarray) -> np.ndarray:
        self.predict()
        return self.update(z)

    def run(self, measurements: np.ndarray) -> np.ndarray:
        measurements = np.asarray(measurements, dtype=float)
        T = measurements.shape[0]
        n = len(self._x)
        states = np.empty((T, n))
        for t in range(T):
            states[t] = self.step(measurements[t])
        return states


def ConstantVelocityKF(
    dt: float,
    process_noise_std: float,
    measurement_noise_std: float,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
) -> KalmanFilter:
    """Create a constant-velocity Kalman filter for range-only tracking.

    State: ``[range, range_rate]``.  Measurement: ``[range]``.
    """
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])

    q = process_noise_std**2
    Q = q * np.array([
        [dt**4 / 4, dt**3 / 2],
        [dt**3 / 2, dt**2],
    ])

    R = np.array([[measurement_noise_std**2]])

    return KalmanFilter(F, H, Q, R, x0=x0, P0=P0)
