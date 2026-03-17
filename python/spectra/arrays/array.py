# python/spectra/arrays/array.py
"""AntennaArray: geometry and steering vector computation."""

from typing import List, Optional, Union

import numpy as np

from spectra.antennas.base import AntennaElement
from spectra.antennas.isotropic import IsotropicElement


class AntennaArray:
    """Planar antenna array with arbitrary element positions.

    Args:
        positions: Element positions in wavelengths, shape (N, 2) for
            (x, y) in the horizontal plane.
        elements: A single AntennaElement (broadcast to all) or a
            list of per-element instances of length N.
        reference_frequency: Frequency in Hz that defines the wavelength for
            the position units. Usually matches the element design frequency.

    Example::

        arr = AntennaArray(
            positions=np.array([[0, 0], [0.5, 0], [1.0, 0]]),
            elements=IsotropicElement(frequency=2.4e9),
            reference_frequency=2.4e9,
        )
        sv = arr.steering_vector(azimuth=np.deg2rad(30), elevation=0.0)
    """

    def __init__(
        self,
        positions: np.ndarray,
        elements: Union[AntennaElement, List[AntennaElement]],
        reference_frequency: float,
    ):
        self.positions = np.asarray(positions, dtype=float)  # (N, 2)
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError("positions must have shape (N, 2)")
        self.reference_frequency = reference_frequency
        n = self.positions.shape[0]
        if isinstance(elements, list):
            if len(elements) != n:
                raise ValueError(
                    f"elements list length {len(elements)} != num_elements {n}"
                )
            self.elements = elements
        else:
            # Broadcast single element to all
            self.elements = [elements] * n

    @property
    def num_elements(self) -> int:
        """Number of array elements."""
        return self.positions.shape[0]

    def steering_vector(
        self,
        azimuth: Union[float, np.ndarray],
        elevation: Union[float, np.ndarray],
    ) -> np.ndarray:
        """Compute the array manifold vector for one or more directions.

        Combines geometry-induced phase shifts with per-element patterns::

            a_i(az, el) = g_i(az, el) * exp(j*2*pi*(x_i*cos(el)*cos(az)
                                                     + y_i*cos(el)*sin(az)))

        Positions are in wavelengths relative to reference_frequency.

        Args:
            azimuth: Azimuth angle(s) in radians. Scalar or 1-D array of length M.
            elevation: Elevation angle(s) in radians. Scalar or 1-D array of length M.

        Returns:
            Complex array of shape (N_elements,) for a single direction or
            (N_elements, M) for multiple directions.
        """
        scalar_input = np.isscalar(azimuth) and np.isscalar(elevation)
        azimuth = np.atleast_1d(np.asarray(azimuth, dtype=float))
        elevation = np.atleast_1d(np.asarray(elevation, dtype=float))
        if azimuth.shape != elevation.shape:
            raise ValueError("azimuth and elevation must have the same shape")
        M = azimuth.size  # number of directions

        x = self.positions[:, 0]  # (N,)
        y = self.positions[:, 1]  # (N,)
        cos_el = np.cos(elevation)  # (M,)
        cos_az = np.cos(azimuth)    # (M,)
        sin_az = np.sin(azimuth)    # (M,)

        # phase_arg[i, m] = x_i * cos(el_m) * cos(az_m) + y_i * cos(el_m) * sin(az_m)
        phase_arg = (
            x[:, np.newaxis] * (cos_el * cos_az)[np.newaxis, :]
            + y[:, np.newaxis] * (cos_el * sin_az)[np.newaxis, :]
        )  # (N, M)
        phase = np.exp(1j * 2 * np.pi * phase_arg)  # (N, M)

        # Pattern component: (N, M)
        pattern = np.zeros((self.num_elements, M), dtype=complex)
        for i, elem in enumerate(self.elements):
            pattern[i, :] = elem.pattern(azimuth, elevation)

        sv = pattern * phase  # (N, M)

        if scalar_input:
            return sv[:, 0]  # (N,)
        return sv  # (N, M)


def ula(
    num_elements: int,
    spacing: float = 0.5,
    element: Optional[AntennaElement] = None,
    frequency: float = 1e9,
) -> AntennaArray:
    """Uniform Linear Array along the x-axis.

    Args:
        num_elements: Number of array elements N.
        spacing: Inter-element spacing in wavelengths. Default 0.5 (lambda/2).
        element: Antenna element instance. Defaults to IsotropicElement.
        frequency: Design frequency in Hz.

    Returns:
        AntennaArray with elements at (n*spacing, 0).
    """
    if element is None:
        element = IsotropicElement(frequency=frequency)
    positions = np.column_stack([
        np.arange(num_elements) * spacing,
        np.zeros(num_elements),
    ])
    return AntennaArray(positions=positions, elements=element, reference_frequency=frequency)


def uca(
    num_elements: int,
    radius: Optional[float] = None,
    element: Optional[AntennaElement] = None,
    frequency: float = 1e9,
) -> AntennaArray:
    """Uniform Circular Array.

    Args:
        num_elements: Number of array elements N.
        radius: Array radius in wavelengths. If None, defaults to the radius
            that gives approximately lambda/2 inter-element spacing:
            radius = 0.5 / (2 * sin(pi / N)).
        element: Antenna element instance. Defaults to IsotropicElement.
        frequency: Design frequency in Hz.

    Returns:
        AntennaArray with elements on a circle in the xy-plane.
    """
    if element is None:
        element = IsotropicElement(frequency=frequency)
    if radius is None:
        radius = 0.5 / (2 * np.sin(np.pi / num_elements))
    angles = 2 * np.pi * np.arange(num_elements) / num_elements
    positions = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    return AntennaArray(positions=positions, elements=element, reference_frequency=frequency)


def rectangular(
    rows: int,
    cols: int,
    spacing_x: float = 0.5,
    spacing_y: float = 0.5,
    element: Optional[AntennaElement] = None,
    frequency: float = 1e9,
) -> AntennaArray:
    """Rectangular planar array.

    Args:
        rows: Number of rows (along y-axis).
        cols: Number of columns (along x-axis).
        spacing_x: Column spacing in wavelengths.
        spacing_y: Row spacing in wavelengths.
        element: Antenna element instance. Defaults to IsotropicElement.
        frequency: Design frequency in Hz.

    Returns:
        AntennaArray with rows * cols elements in a grid.
    """
    if element is None:
        element = IsotropicElement(frequency=frequency)
    xs = np.arange(cols) * spacing_x
    ys = np.arange(rows) * spacing_y
    grid_x, grid_y = np.meshgrid(xs, ys)
    positions = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    return AntennaArray(positions=positions, elements=element, reference_frequency=frequency)
