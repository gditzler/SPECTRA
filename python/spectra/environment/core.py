"""Core Environment, Emitter, ReceiverConfig, and LinkParams classes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from spectra.environment.position import Position
from spectra.environment.propagation import PropagationModel
from spectra.waveforms.base import Waveform

SPEED_OF_LIGHT = 299_792_458.0
BOLTZMANN_K = 1.380649e-23  # J/K


@dataclass
class Emitter:
    """A transmitting source with waveform, position, and RF parameters."""

    waveform: Waveform
    position: Position
    power_dbm: float
    freq_hz: float
    velocity_mps: tuple[float, float] | None = None
    antenna_gain_dbi: float = 0.0


@dataclass
class ReceiverConfig:
    """Receiver parameters for link budget computation."""

    position: Position
    noise_figure_db: float = 6.0
    bandwidth_hz: float = 1e6
    antenna_gain_dbi: float = 0.0
    temperature_k: float = 290.0


@dataclass
class LinkParams:
    """Derived link parameters for a single emitter."""

    emitter_index: int
    snr_db: float
    path_loss_db: float
    received_power_dbm: float
    delay_s: float
    doppler_hz: float
    distance_m: float
    fading_suggestion: str | None


class Environment:
    """Computes per-emitter link parameters from geometry and propagation."""

    def __init__(
        self,
        propagation: PropagationModel,
        emitters: list[Emitter],
        receiver: ReceiverConfig,
    ):
        self.propagation = propagation
        self.emitters = emitters
        self.receiver = receiver

    def compute(self, seed: int | None = None) -> list[LinkParams]:
        """Compute link parameters for each emitter."""
        results = []
        for i, emitter in enumerate(self.emitters):
            distance = emitter.position.distance_to(self.receiver.position)

            # Propagation model — derive per-emitter seed from master seed
            kwargs = {}
            if seed is not None:
                kwargs["seed"] = seed + i
            pl_result = self.propagation(distance, emitter.freq_hz, **kwargs)

            # Link budget
            rx_power = (
                emitter.power_dbm
                + emitter.antenna_gain_dbi
                + self.receiver.antenna_gain_dbi
                - pl_result.path_loss_db
            )
            noise_power = (
                10 * math.log10(BOLTZMANN_K * self.receiver.temperature_k)
                + 30  # convert to dBm
                + 10 * math.log10(self.receiver.bandwidth_hz)
                + self.receiver.noise_figure_db
            )
            snr = rx_power - noise_power

            # Propagation delay
            delay = distance / SPEED_OF_LIGHT

            # Doppler
            doppler = 0.0
            if emitter.velocity_mps is not None:
                bearing = self.receiver.position.bearing_to(emitter.position)
                vx, vy = emitter.velocity_mps
                v_radial = -(vx * math.cos(bearing) + vy * math.sin(bearing))
                doppler = (v_radial / SPEED_OF_LIGHT) * emitter.freq_hz

            # Fading suggestion from propagation model metadata
            fading = None
            if pl_result.k_factor_db is not None:
                fading = f"rician_k{int(pl_result.k_factor_db)}"
            elif pl_result.rms_delay_spread_s is not None:
                fading = "rayleigh"

            results.append(
                LinkParams(
                    emitter_index=i,
                    snr_db=snr,
                    path_loss_db=pl_result.path_loss_db,
                    received_power_dbm=rx_power,
                    delay_s=delay,
                    doppler_hz=doppler,
                    distance_m=distance,
                    fading_suggestion=fading,
                )
            )
        return results
