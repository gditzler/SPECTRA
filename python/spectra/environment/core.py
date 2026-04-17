"""Core Environment, Emitter, ReceiverConfig, and LinkParams classes."""

from __future__ import annotations

import math
from dataclasses import dataclass

from spectra.environment.position import Position
from spectra.environment.propagation import (
    ITU_R_P525,
    ITU_R_P1411,
    COST231HataPL,
    FreeSpacePathLoss,
    GPP38901InH,
    GPP38901RMa,
    GPP38901UMa,
    GPP38901UMi,
    LogDistancePL,
    OkumuraHataPL,
    PropagationModel,
)
from spectra.waveforms.base import Waveform

_PROPAGATION_REGISTRY: dict[str, type[PropagationModel]] = {
    "free_space": FreeSpacePathLoss,
    "log_distance": LogDistancePL,
    "cost231_hata": COST231HataPL,
    "itu_r_p525": ITU_R_P525,
    "okumura_hata": OkumuraHataPL,
    "gpp_38_901_uma": GPP38901UMa,
    "gpp_38_901_umi": GPP38901UMi,
    "gpp_38_901_rma": GPP38901RMa,
    "gpp_38_901_inh": GPP38901InH,
    "itu_r_p1411": ITU_R_P1411,
}


def _waveform_to_dict(waveform: Waveform) -> dict:
    """Serialize a waveform to a type/params dict."""
    cls_name = type(waveform).__name__
    params = {}
    if hasattr(waveform, "__dict__"):
        for k, v in waveform.__dict__.items():
            if not k.startswith("_"):
                params[k] = v
    return {"type": cls_name, "params": params} if params else {"type": cls_name}


def _waveform_from_dict(d: dict) -> Waveform:
    """Deserialize a waveform from a type/params dict."""
    import spectra.waveforms as wmod

    cls = getattr(wmod, d["type"])
    params = d.get("params", {})
    return cls(**params)


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
    # Populated from PathLossResult (optional; defaulted for back-compat)
    shadow_fading_db: float = 0.0
    rms_delay_spread_s: float | None = None
    k_factor_db: float | None = None
    angular_spread_deg: float | None = None


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
                    shadow_fading_db=pl_result.shadow_fading_db,
                    rms_delay_spread_s=pl_result.rms_delay_spread_s,
                    k_factor_db=pl_result.k_factor_db,
                    angular_spread_deg=pl_result.angular_spread_deg,
                )
            )
        return results

    def to_yaml(self, path: str) -> None:
        """Serialize this Environment to a YAML file."""
        import yaml

        prop = self.propagation
        # Reverse-lookup registry key for this propagation instance
        registry_key: str | None = None
        for k, cls in _PROPAGATION_REGISTRY.items():
            if type(prop) is cls:
                registry_key = k
                break
        if registry_key is None:
            raise ValueError(f"Unknown propagation type {type(prop).__name__}; not in registry")

        # Collect constructor params from public attrs (matches __init__ signature)
        prop_dict: dict = {"type": registry_key}
        for name in vars(prop):
            if name.startswith("_"):
                continue
            prop_dict[name] = getattr(prop, name)

        emitters_list = []
        for e in self.emitters:
            entry: dict = {
                "waveform": _waveform_to_dict(e.waveform),
                "position": [e.position.x, e.position.y]
                + ([e.position.z] if e.position.z is not None else []),
                "power_dbm": e.power_dbm,
                "freq_hz": e.freq_hz,
            }
            if e.velocity_mps is not None:
                entry["velocity_mps"] = list(e.velocity_mps)
            if e.antenna_gain_dbi != 0.0:
                entry["antenna_gain_dbi"] = e.antenna_gain_dbi
            emitters_list.append(entry)

        rx = self.receiver
        rx_dict: dict = {
            "position": [rx.position.x, rx.position.y]
            + ([rx.position.z] if rx.position.z is not None else []),
            "noise_figure_db": rx.noise_figure_db,
            "bandwidth_hz": rx.bandwidth_hz,
        }
        if rx.antenna_gain_dbi != 0.0:
            rx_dict["antenna_gain_dbi"] = rx.antenna_gain_dbi
        if rx.temperature_k != 290.0:
            rx_dict["temperature_k"] = rx.temperature_k

        data = {
            "environment": {
                "propagation": prop_dict,
                "receiver": rx_dict,
                "emitters": emitters_list,
            }
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "Environment":
        """Deserialize an Environment from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        env_data = data["environment"]

        # Propagation model
        prop_data = env_data["propagation"]
        prop_type = prop_data["type"]
        prop_cls = _PROPAGATION_REGISTRY[prop_type]
        prop_params = {k: v for k, v in prop_data.items() if k != "type"}
        propagation = prop_cls(**prop_params)

        # Receiver
        rx_data = env_data["receiver"]
        rx_pos_list = rx_data["position"]
        rx_pos = Position(
            rx_pos_list[0],
            rx_pos_list[1],
            rx_pos_list[2] if len(rx_pos_list) > 2 else None,
        )
        receiver = ReceiverConfig(
            position=rx_pos,
            noise_figure_db=rx_data.get("noise_figure_db", 6.0),
            bandwidth_hz=rx_data.get("bandwidth_hz", 1e6),
            antenna_gain_dbi=rx_data.get("antenna_gain_dbi", 0.0),
            temperature_k=rx_data.get("temperature_k", 290.0),
        )

        # Emitters
        emitters = []
        for e_data in env_data["emitters"]:
            pos_list = e_data["position"]
            pos = Position(
                pos_list[0],
                pos_list[1],
                pos_list[2] if len(pos_list) > 2 else None,
            )
            vel = tuple(e_data["velocity_mps"]) if "velocity_mps" in e_data else None
            emitters.append(
                Emitter(
                    waveform=_waveform_from_dict(e_data["waveform"]),
                    position=pos,
                    power_dbm=e_data["power_dbm"],
                    freq_hz=e_data["freq_hz"],
                    velocity_mps=vel,
                    antenna_gain_dbi=e_data.get("antenna_gain_dbi", 0.0),
                )
            )

        return cls(propagation=propagation, emitters=emitters, receiver=receiver)
