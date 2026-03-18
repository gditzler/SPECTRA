"""Target trajectory motion models for radar simulation.

Two motion models are provided:

- :class:`ConstantVelocity` — linear range propagation at fixed velocity.
- :class:`ConstantTurnRate` — 1-D range projection of a 2-D circular arc.

Both satisfy the :class:`Trajectory` protocol and can be used interchangeably
in :class:`~spectra.datasets.radar_pipeline.RadarPipelineDataset`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Trajectory(Protocol):
    """Protocol for target trajectory models."""

    def state_at(self, step: int) -> np.ndarray: ...
    def states(self, num_steps: int) -> np.ndarray: ...
    def range_at(self, step: int) -> float: ...


class ConstantVelocity:
    """Constant-velocity (linear) trajectory.

    State vector: ``[range, range_rate]``.

    Args:
        initial_range: Starting range in metres.
        velocity: Constant range rate in m/s (positive = opening).
        dt: Time step between trajectory steps in seconds.
    """

    def __init__(self, initial_range: float, velocity: float, dt: float) -> None:
        self.initial_range = initial_range
        self.velocity = velocity
        self.dt = dt

    def state_at(self, step: int) -> np.ndarray:
        t = step * self.dt
        return np.array([self.initial_range + self.velocity * t, self.velocity])

    def states(self, num_steps: int) -> np.ndarray:
        out = np.empty((num_steps, 2))
        for i in range(num_steps):
            out[i] = self.state_at(i)
        return out

    def range_at(self, step: int) -> float:
        return float(self.state_at(step)[0])


class ConstantTurnRate:
    """Constant-turn-rate trajectory (1-D range projection of 2-D turning).

    Models a target moving at constant speed on a circular arc, observed
    from a fixed radar at the origin. The target starts at ``(initial_range, 0)``.

    Propagation::

        x(t) = x0 + (v / omega) * sin(omega * t)
        y(t) = (v / omega) * (1 - cos(omega * t))
        range(t) = sqrt(x(t)^2 + y(t)^2)

    State vector: ``[range, range_rate]``.

    Args:
        initial_range: Starting range in metres.
        velocity: Constant speed in m/s.
        turn_rate: Turn rate in rad/s.
        dt: Time step between trajectory steps in seconds.
    """

    def __init__(
        self, initial_range: float, velocity: float, turn_rate: float, dt: float
    ) -> None:
        self.initial_range = initial_range
        self.velocity = velocity
        self.turn_rate = turn_rate
        self.dt = dt

    def _xy(self, t: float):
        omega = self.turn_rate
        v = self.velocity
        if abs(omega) < 1e-12:
            return self.initial_range + v * t, 0.0
        x = self.initial_range + (v / omega) * np.sin(omega * t)
        y = (v / omega) * (1.0 - np.cos(omega * t))
        return float(x), float(y)

    def state_at(self, step: int) -> np.ndarray:
        t = step * self.dt
        x, y = self._xy(t)
        r = np.sqrt(x**2 + y**2)
        eps = self.dt * 0.001
        x2, y2 = self._xy(t + eps)
        r2 = np.sqrt(x2**2 + y2**2)
        rdot = (r2 - r) / eps
        return np.array([r, rdot])

    def states(self, num_steps: int) -> np.ndarray:
        out = np.empty((num_steps, 2))
        for i in range(num_steps):
            out[i] = self.state_at(i)
        return out

    def range_at(self, step: int) -> float:
        return float(self.state_at(step)[0])
