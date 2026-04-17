"""Pure-logic config builder for the Signal Builder CLI.

Provides registries, categories, presets, config construction,
validation, and YAML serialization — all without interactive I/O.
"""

from typing import Any, Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

_WAVEFORM_REGISTRY: Optional[Dict[str, type]] = None
_IMPAIRMENT_REGISTRY: Optional[Dict[str, type]] = None

_IMPAIRMENT_EXCLUDES = {
    "Compose",
    "exponential_correlation",
    "kronecker_correlation",
    "steering_vector",
}

# Names exported from spectra.waveforms that are support infrastructure, not
# selectable waveforms (ABCs, dataclasses, helper functions). Excluded from
# the interactive signal-pool builder so users don't try to pick a Schedule
# or a dataclass as a waveform type.
_WAVEFORM_EXCLUDES = {
    "Schedule",
    "StaticSchedule",
    "StochasticSchedule",
    "CognitiveSchedule",
    "SegmentSpec",
    "ModeSpec",
    "segments_to_mode_mask",
}


def get_waveform_registry() -> Dict[str, type]:
    global _WAVEFORM_REGISTRY
    if _WAVEFORM_REGISTRY is None:
        from spectra import waveforms as wmod

        _WAVEFORM_REGISTRY = {
            name: getattr(wmod, name) for name in wmod.__all__ if name not in _WAVEFORM_EXCLUDES
        }
    return _WAVEFORM_REGISTRY


def get_impairment_registry() -> Dict[str, type]:
    global _IMPAIRMENT_REGISTRY
    if _IMPAIRMENT_REGISTRY is None:
        from spectra import impairments as imod

        _IMPAIRMENT_REGISTRY = {
            name: getattr(imod, name) for name in imod.__all__ if name not in _IMPAIRMENT_EXCLUDES
        }
    return _IMPAIRMENT_REGISTRY


# ---------------------------------------------------------------------------
# Waveform categories
# ---------------------------------------------------------------------------

WAVEFORM_CATEGORIES: Dict[str, List[str]] = {
    "PSK": ["BPSK", "QPSK", "PSK8", "PSK16", "PSK32", "PSK64"],
    "QAM": ["QAM16", "QAM32", "QAM64", "QAM128", "QAM256", "QAM512", "QAM1024"],
    "FSK": [
        "FSK",
        "FSK4",
        "FSK8",
        "FSK16",
        "GFSK",
        "GFSK4",
        "GFSK8",
        "GFSK16",
        "GMSK",
        "GMSK4",
        "GMSK8",
        "MSK",
        "MSK4",
        "MSK8",
    ],
    "ASK": ["OOK", "ASK4", "ASK8", "ASK16", "ASK32", "ASK64"],
    "OFDM": [
        "OFDM",
        "OFDM72",
        "OFDM128",
        "OFDM180",
        "OFDM256",
        "OFDM300",
        "OFDM512",
        "OFDM600",
        "OFDM900",
        "OFDM1200",
        "OFDM2048",
        "SCFDMA",
    ],
    "Analog": ["AMDSB", "AMDSB_SC", "AMLSB", "AMUSB", "FM", "Tone"],
    "Radar": [
        "LFM",
        "BarkerCode",
        "FrankCode",
        "P1Code",
        "P2Code",
        "P3Code",
        "P4Code",
        "CostasCode",
        "PulsedRadar",
        "BarkerCodedPulse",
        "PolyphaseCodedPulse",
        "FMCW",
        "NonlinearFM",
        "SteppedFrequency",
        "PulseDoppler",
    ],
    "5G NR": ["NR_OFDM", "NR_PDSCH", "NR_PRACH", "NR_PUSCH", "NR_SSB"],
    "Protocol": ["ACARS", "ADSB", "AIS", "DME", "ILS_Localizer", "ModeS"],
    "Spread Spectrum": [
        "DSSS_BPSK",
        "DSSS_QPSK",
        "FHSS",
        "THSS",
        "CDMA_Forward",
        "CDMA_Reverse",
        "ChirpSS",
    ],
    "Other": ["Noise", "ZadoffChu"],
    "Multi-function Emitter": [
        "ScheduledWaveform",
        "multifunction_search_track_radar",
        "multi_prf_pulse_doppler_radar",
        "frequency_agile_stepped_pri_radar",
        "radcom_emitter",
    ],
}

# ---------------------------------------------------------------------------
# Impairment presets
# ---------------------------------------------------------------------------

IMPAIRMENT_PRESETS: Dict[str, List[Dict[str, Any]]] = {
    "clean": [
        {"type": "AWGN", "params": {"snr_range": [-5, 30]}},
    ],
    "mild": [
        {"type": "FrequencyOffset", "params": {"max_offset": 50_000}},
        {"type": "PhaseOffset", "params": {"max_offset": 3.14159}},
        {"type": "AWGN", "params": {"snr_range": [-5, 30]}},
    ],
    "realistic": [
        {"type": "FrequencyOffset", "params": {"max_offset": 50_000}},
        {"type": "PhaseOffset", "params": {"max_offset": 3.14159}},
        {"type": "IQImbalance"},
        {"type": "DCOffset", "params": {"max_offset": 0.05}},
        {"type": "PhaseNoise"},
        {"type": "AWGN", "params": {"snr_range": [-5, 30]}},
    ],
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def default_seeds() -> Dict[str, int]:
    return {"train": 1000, "val": 2000, "test": 3000}


def default_num_samples() -> Dict[str, int]:
    return {"train": 50_000, "val": 10_000, "test": 10_000}


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------


def build_config(
    *,
    name: str,
    task: str,
    waveform_pool: List[Dict[str, Any]],
    impairments: List[Dict[str, Any]],
    snr_range: Tuple[float, float],
    sample_rate: float,
    num_iq_samples: int,
    num_samples: Dict[str, int],
    seed: Dict[str, int],
    scene: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a benchmark-compatible config dict."""
    cfg: Dict[str, Any] = {
        "name": name,
        "version": "1.0",
        "task": task,
        "sample_rate": sample_rate,
        "num_iq_samples": num_iq_samples,
        "num_samples": dict(num_samples),
        "seed": dict(seed),
        "waveform_pool": list(waveform_pool),
        "snr_range": list(snr_range),
        "impairments": list(impairments),
    }
    if task == "wideband" and scene is not None:
        cfg["scene"] = dict(scene)
    return cfg


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_config(cfg: Dict[str, Any]) -> List[str]:
    """Return a list of error strings (empty == valid)."""
    errors: List[str] = []
    wreg = get_waveform_registry()
    ireg = get_impairment_registry()

    for entry in cfg.get("waveform_pool", []):
        wtype = entry.get("type", "")
        if wtype not in wreg:
            errors.append(f"Unknown waveform type: '{wtype}'")

    for entry in cfg.get("impairments", []):
        itype = entry.get("type", "")
        if itype not in ireg:
            errors.append(f"Unknown impairment type: '{itype}'")

    snr = cfg.get("snr_range", [])
    if len(snr) != 2:
        errors.append(f"snr_range must have exactly 2 elements, got {len(snr)}")

    for key in ("name", "task", "sample_rate", "num_iq_samples", "num_samples", "seed"):
        if key not in cfg:
            errors.append(f"Missing required key: '{key}'")

    return errors


# ---------------------------------------------------------------------------
# YAML serialization
# ---------------------------------------------------------------------------


class _FlowList(list):
    """List subclass that YAML dumps in flow (inline) style."""


class _FlowDict(dict):
    """Dict subclass that YAML dumps in flow (inline) style."""


def _flow_representer(dumper: yaml.SafeDumper, data: _FlowList) -> yaml.Node:
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def _flow_dict_representer(dumper: yaml.SafeDumper, data: _FlowDict) -> yaml.Node:
    return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=True)


def _prepare_for_yaml(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Convert waveform_pool and impairments entries to flow-style dicts."""
    out = dict(cfg)
    out["waveform_pool"] = [_FlowDict(e) for e in cfg.get("waveform_pool", [])]
    out["impairments"] = [_FlowDict(e) for e in cfg.get("impairments", [])]
    out["snr_range"] = _FlowList(cfg.get("snr_range", []))
    if "scene" in cfg:
        scene = dict(cfg["scene"])
        if "num_signals" in scene:
            scene["num_signals"] = _FlowList(scene["num_signals"])
        out["scene"] = scene
    return out


def serialize_config(cfg: Dict[str, Any]) -> str:
    """Serialize a config dict to a YAML string."""
    dumper = yaml.SafeDumper
    dumper.add_representer(_FlowList, _flow_representer)
    dumper.add_representer(_FlowDict, _flow_dict_representer)
    prepared = _prepare_for_yaml(cfg)
    return yaml.dump(prepared, Dumper=dumper, default_flow_style=False, sort_keys=False)
