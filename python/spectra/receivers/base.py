"""Receiver and Decoder abstract base classes with FEC stubs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Receiver(ABC):
    """Abstract base class for receivers.

    Receivers map IQ samples to symbol indices and bits. Unlike
    :class:`~spectra.impairments.base.Transform` (which maps IQ to IQ),
    receivers output a fundamentally different domain.
    """

    @abstractmethod
    def demodulate(self, received_iq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Demodulate received IQ samples.

        Returns:
            Tuple of ``(symbol_indices, bits)`` as uint32/uint8 arrays.
        """


class Decoder(ABC):
    """Abstract base class for FEC decoders."""

    @abstractmethod
    def decode(self, bits: np.ndarray) -> np.ndarray:
        """Decode a block of (possibly coded) bits. Returns information bits."""


class PassthroughDecoder(Decoder):
    """No-op decoder. Returns input unchanged."""

    def decode(self, bits: np.ndarray) -> np.ndarray:
        return bits


class ViterbiDecoder(Decoder):
    """Convolutional code decoder stub.

    Args:
        constraint_length: Constraint length of the convolutional code.
        generators: Generator polynomials as integers.
    """

    def __init__(self, constraint_length: int, generators: list) -> None:
        self.constraint_length = constraint_length
        self.generators = generators

    def decode(self, bits: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "ViterbiDecoder is a stub. Implement decode() for your code."
        )


class LDPCDecoder(Decoder):
    """LDPC decoder stub.

    Args:
        parity_check_matrix: Parity check matrix H.
        max_iterations: Maximum belief propagation iterations.
    """

    def __init__(self, parity_check_matrix: np.ndarray, max_iterations: int = 50) -> None:
        self.parity_check_matrix = parity_check_matrix
        self.max_iterations = max_iterations

    def decode(self, bits: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "LDPCDecoder is a stub. Implement decode() for your code."
        )
