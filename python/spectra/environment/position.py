"""Position dataclass with 2D/3D geometry helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Position:
    """A point in 2D (or optionally 3D) space. All units in meters."""

    x: float
    y: float
    z: float | None = None

    def distance_to(self, other: Position) -> float:
        """Euclidean distance. Uses 3D when both positions have z, else 2D."""
        dx = other.x - self.x
        dy = other.y - self.y
        if self.z is not None and other.z is not None:
            dz = other.z - self.z
            return math.sqrt(dx * dx + dy * dy + dz * dz)
        return math.sqrt(dx * dx + dy * dy)

    def bearing_to(self, other: Position) -> float:
        """Azimuth angle from self to other in radians. 0 = +x, pi/2 = +y."""
        dx = other.x - self.x
        dy = other.y - self.y
        return math.atan2(dy, dx)

    def angle_to(self, other: Position) -> float:
        """Azimuth angle from self to other (alias for bearing_to)."""
        return self.bearing_to(other)

    def elevation_to(self, other: Position) -> float | None:
        """Elevation angle in radians. Returns None if either z is None."""
        if self.z is None or other.z is None:
            return None
        horiz = math.sqrt((other.x - self.x) ** 2 + (other.y - self.y) ** 2)
        dz = other.z - self.z
        return math.atan2(dz, horiz)
