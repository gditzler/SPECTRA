from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple


@dataclass
class DatasetMetadata:
    """Base metadata for a dataset."""

    name: str = "spectra_dataset"
    num_samples: int = 1000
    sample_rate: float = 1e6
    seed: int = 42

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "DatasetMetadata":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str) -> None:
        import yaml

        # Convert tuples to lists for safe YAML serialization
        def _sanitize(obj):
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_sanitize(v) for v in obj]
            return obj

        with open(path, "w") as f:
            yaml.dump(_sanitize(self.to_dict()), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str) -> "DatasetMetadata":
        import yaml

        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)


@dataclass
class NarrowbandMetadata(DatasetMetadata):
    """Metadata for narrowband (single-signal classification) datasets."""

    waveform_labels: List[str] = field(default_factory=lambda: ["QPSK", "BPSK"])
    num_iq_samples: int = 1024
    snr_range: Tuple[float, float] = (0.0, 20.0)
    impairment_config: Dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict) -> "NarrowbandMetadata":
        # Convert snr_range from list to tuple if needed
        if "snr_range" in d and isinstance(d["snr_range"], list):
            d = dict(d)
            d["snr_range"] = tuple(d["snr_range"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str) -> "NarrowbandMetadata":
        import yaml

        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)


@dataclass
class WidebandMetadata(DatasetMetadata):
    """Metadata for wideband (multi-signal detection) datasets."""

    capture_bandwidth: float = 1e6
    capture_duration: float = 1e-3
    num_signals_range: Tuple[int, int] = (1, 5)

    @classmethod
    def from_dict(cls, d: Dict) -> "WidebandMetadata":
        if "num_signals_range" in d and isinstance(d["num_signals_range"], list):
            d = dict(d)
            d["num_signals_range"] = tuple(d["num_signals_range"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str) -> "WidebandMetadata":
        import yaml

        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)
