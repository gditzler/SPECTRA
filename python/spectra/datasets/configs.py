"""Dataclass-based configuration objects for SPECTRA datasets.

Each config maps 1-to-1 to a Dataset constructor and can round-trip through
YAML/JSON for reproducibility and hyper-parameter sweeps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

# ---------------------------------------------------------------------------
# Sanitisation helpers
# ---------------------------------------------------------------------------


def _sanitize_for_yaml(obj: Any) -> Any:
    """Recursively convert tuples/lists/dicts to YAML-safe structures."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_yaml(v) for v in obj]
    return obj


def _restore_tuples(obj: Any, fields: Optional[List[str]] = None) -> Any:
    """Restore list-to-tuple only for explicitly named top-level dict keys."""
    if fields is None:
        fields = []
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if isinstance(v, list) and k in fields:
                result[k] = tuple(v)
            else:
                result[k] = v
        return result
    if isinstance(obj, list):
        return [_restore_tuples(v, fields) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_CONFIG_REGISTRY: Dict[str, Type["BaseDatasetConfig"]] = {}


def register_config(name: str):
    """Decorator that registers a config subclass by string key."""

    def decorator(cls: Type["BaseDatasetConfig"]) -> Type["BaseDatasetConfig"]:
        _CONFIG_REGISTRY[name] = cls
        return cls

    return decorator


def resolve_config(name: str) -> Type["BaseDatasetConfig"]:
    """Lookup a registered config class by name."""
    try:
        return _CONFIG_REGISTRY[name]
    except KeyError:
        available = ", ".join(_CONFIG_REGISTRY.keys())
        raise ValueError(
            f"Unknown config type {name!r}. Available: {available}"
        ) from None


# ---------------------------------------------------------------------------
# Base config
# ---------------------------------------------------------------------------


@dataclass
class BaseDatasetConfig:
    """Base configuration shared by all SPECTRA datasets.

    Attributes:
        config_type: Discriminator used during YAML deserialization.
        name: Human-readable identifier.
        num_samples: Total number of samples in the dataset.
        sample_rate: Receiver sample rate in Hz.
        seed: Base integer seed for deterministic generation.
    """

    config_type: str = "base"
    name: str = "spectra_dataset"
    num_samples: int = 1000
    sample_rate: float = 1e6
    seed: int = 42

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Run lightweight validation. Subclasses should extend this."""
        if self.num_samples <= 0:
            raise ValueError(
                f"num_samples must be positive, got {self.num_samples}"
            )
        if self.sample_rate <= 0:
            raise ValueError(
                f"sample_rate must be positive, got {self.sample_rate}"
            )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return _sanitize_for_yaml(asdict(self))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BaseDatasetConfig":
        """Deserialize from a dictionary, ignoring unknown keys."""
        # Identify tuple-range fields by inspecting dataclass annotations
        tuple_keys = [
            k for k, v in cls.__dataclass_fields__.items()
            if hasattr(v, "type") and "Tuple" in str(v.type)
        ]
        d = _restore_tuples(dict(d), tuple_keys)
        filtered = {
            k: v for k, v in d.items() if k in cls.__dataclass_fields__
        }
        return cls(**filtered)

    def to_yaml(self, path: str) -> None:
        """Write to a YAML file."""
        import yaml

        payload = {"config_type": self.config_type, **self.to_dict()}
        with open(path, "w") as f:
            yaml.dump(payload, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "BaseDatasetConfig":
        """Load from a YAML file, dispatching to the correct subclass."""
        import yaml

        with open(path, "r") as f:
            d = yaml.safe_load(f)
        ctype = d.pop("config_type", "base")
        if ctype == "base":
            target_cls = cls
        else:
            target_cls = resolve_config(ctype)
        return target_cls.from_dict(d)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def build_dataset(self, **kwargs: Any) -> Any:
        """Build the corresponding PyTorch Dataset.

        Subclasses must implement this.  ``kwargs`` typically includes
        ``waveform_pool`` or ``signal_pool`` which are not serialisable.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.build_dataset() is not implemented"
        )


# ---------------------------------------------------------------------------
# Narrowband config
# ---------------------------------------------------------------------------


@dataclass
@register_config("narrowband")
class NarrowbandConfig(BaseDatasetConfig):
    """Configuration for :class:`~spectra.datasets.narrowband.NarrowbandDataset`."""

    config_type: str = "narrowband"
    num_iq_samples: int = 1024
    snr_range: Optional[Tuple[float, float]] = None
    class_weights: Optional[List[float]] = None
    mimo_config: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        super().validate()
        if self.num_iq_samples <= 0:
            raise ValueError(
                f"num_iq_samples must be positive, got {self.num_iq_samples}"
            )
        if self.snr_range is not None and len(self.snr_range) != 2:
            raise ValueError(
                f"snr_range must be a 2-tuple, got {self.snr_range}"
            )
        if self.class_weights is not None and len(self.class_weights) == 0:
            raise ValueError("class_weights must not be empty")

    def build_dataset(self, waveform_pool, impairments=None, transform=None):
        """Build a :class:`~spectra.datasets.narrowband.NarrowbandDataset`.

        Args:
            waveform_pool: List of :class:`~spectra.waveforms.base.Waveform`.
            impairments: Optional :class:`~spectra.impairments.compose.Compose`.
            transform: Optional transform callable.

        Returns:
            An instantiated ``NarrowbandDataset``.
        """
        from spectra.datasets.narrowband import NarrowbandDataset

        return NarrowbandDataset(
            waveform_pool=waveform_pool,
            num_samples=self.num_samples,
            num_iq_samples=self.num_iq_samples,
            sample_rate=self.sample_rate,
            impairments=impairments,
            transform=transform,
            seed=self.seed,
            class_weights=self.class_weights,
            mimo_config=self.mimo_config,
        )


# ---------------------------------------------------------------------------
# Wideband config
# ---------------------------------------------------------------------------


@dataclass
@register_config("wideband")
class WidebandConfig(BaseDatasetConfig):
    """Configuration for :class:`~spectra.datasets.wideband.WidebandDataset`."""

    config_type: str = "wideband"
    capture_bandwidth: float = 1e6
    capture_duration: float = 1e-3
    num_signals: Union[int, Tuple[int, int]] = (1, 5)
    snr_range: Tuple[float, float] = (5.0, 20.0)
    allow_overlap: bool = True

    def validate(self) -> None:
        super().validate()
        if self.capture_bandwidth <= 0:
            raise ValueError(
                f"capture_bandwidth must be positive, got {self.capture_bandwidth}"
            )
        if self.capture_duration <= 0:
            raise ValueError(
                f"capture_duration must be positive, got {self.capture_duration}"
            )
        if isinstance(self.num_signals, tuple):
            if (
                len(self.num_signals) != 2
                or self.num_signals[0] > self.num_signals[1]
                or self.num_signals[0] < 0
            ):
                raise ValueError(
                    f"num_signals tuple must be (min, max) with 0 <= min <= max, "
                    f"got {self.num_signals}"
                )
        elif self.num_signals <= 0:
            raise ValueError(
                f"num_signals must be positive, got {self.num_signals}"
            )

    def build_dataset(self, signal_pool, impairments=None, transform=None):
        """Build a :class:`~spectra.datasets.wideband.WidebandDataset`.

        Args:
            signal_pool: List of :class:`~spectra.waveforms.base.Waveform`.
            impairments: Optional :class:`~spectra.impairments.compose.Compose`.
            transform: Optional transform callable.

        Returns:
            An instantiated ``WidebandDataset``.
        """
        from spectra.datasets.wideband import WidebandDataset
        from spectra.scene.composer import SceneConfig

        scene_config = SceneConfig(
            capture_duration=self.capture_duration,
            capture_bandwidth=self.capture_bandwidth,
            sample_rate=self.sample_rate,
            num_signals=self.num_signals,
            signal_pool=signal_pool,
            snr_range=self.snr_range,
            allow_overlap=self.allow_overlap,
        )
        return WidebandDataset(
            scene_config=scene_config,
            num_samples=self.num_samples,
            impairments=impairments,
            transform=transform,
            seed=self.seed,
        )


# ---------------------------------------------------------------------------
# Radar config
# ---------------------------------------------------------------------------


@dataclass
@register_config("radar")
class RadarConfig(BaseDatasetConfig):
    """Configuration for :class:`~spectra.datasets.radar.RadarDataset`."""

    config_type: str = "radar"
    num_range_bins: int = 512
    snr_range: Tuple[float, float] = (5.0, 25.0)
    num_targets_range: Tuple[int, int] = (1, 3)

    def validate(self) -> None:
        super().validate()
        if self.num_range_bins <= 0:
            raise ValueError(
                f"num_range_bins must be positive, got {self.num_range_bins}"
            )
        if len(self.snr_range) != 2:
            raise ValueError(
                f"snr_range must be a 2-tuple, got {self.snr_range}"
            )
        if (
            len(self.num_targets_range) != 2
            or self.num_targets_range[0] < 0
            or self.num_targets_range[0] > self.num_targets_range[1]
        ):
            raise ValueError(
                f"num_targets_range must be (min, max) with 0 <= min <= max, "
                f"got {self.num_targets_range}"
            )

    def build_dataset(self, waveform_pool):
        """Build a :class:`~spectra.datasets.radar.RadarDataset`.

        Args:
            waveform_pool: List of :class:`~spectra.waveforms.base.Waveform`.

        Returns:
            An instantiated ``RadarDataset``.
        """
        from spectra.datasets.radar import RadarDataset

        return RadarDataset(
            waveform_pool=waveform_pool,
            num_range_bins=self.num_range_bins,
            sample_rate=self.sample_rate,
            snr_range=self.snr_range,
            num_targets_range=self.num_targets_range,
            num_samples=self.num_samples,
            seed=self.seed,
        )


# ---------------------------------------------------------------------------
# SNR sweep config
# ---------------------------------------------------------------------------


@dataclass
@register_config("snr_sweep")
class SNRSweepConfig(BaseDatasetConfig):
    """Configuration for :class:`~spectra.datasets.snr_sweep.SNRSweepDataset`."""

    config_type: str = "snr_sweep"
    num_iq_samples: int = 1024
    snr_levels: List[float] = field(default_factory=lambda: [-10.0, 0.0, 10.0, 20.0])
    samples_per_cell: int = 500

    def validate(self) -> None:
        super().validate()
        if self.num_iq_samples <= 0:
            raise ValueError(
                f"num_iq_samples must be positive, got {self.num_iq_samples}"
            )
        if len(self.snr_levels) == 0:
            raise ValueError("snr_levels must not be empty")
        if self.samples_per_cell <= 0:
            raise ValueError(
                f"samples_per_cell must be positive, got {self.samples_per_cell}"
            )

    def build_dataset(self, waveform_pool, impairments_fn):
        """Build a :class:`~spectra.datasets.snr_sweep.SNRSweepDataset`.

        Args:
            waveform_pool: List of :class:`~spectra.waveforms.base.Waveform`.
            impairments_fn: Callable ``snr_db -> Compose``.

        Returns:
            An instantiated ``SNRSweepDataset``.
        """
        from spectra.datasets.snr_sweep import SNRSweepDataset

        return SNRSweepDataset(
            waveform_pool=waveform_pool,
            snr_levels=self.snr_levels,
            samples_per_cell=self.samples_per_cell,
            num_iq_samples=self.num_iq_samples,
            sample_rate=self.sample_rate,
            impairments_fn=impairments_fn,
            seed=self.seed,
        )


# ---------------------------------------------------------------------------
# Direction-finding config
# ---------------------------------------------------------------------------


@dataclass
@register_config("direction_finding")
class DirectionFindingConfig(BaseDatasetConfig):
    """Configuration for
    :class:`~spectra.datasets.direction_finding.DirectionFindingDataset`."""

    config_type: str = "direction_finding"
    num_signals: int = 2
    num_snapshots: int = 128
    snr_range: Tuple[float, float] = (10.0, 20.0)

    def validate(self) -> None:
        super().validate()
        if self.num_signals <= 0:
            raise ValueError(
                f"num_signals must be positive, got {self.num_signals}"
            )
        if self.num_snapshots <= 0:
            raise ValueError(
                f"num_snapshots must be positive, got {self.num_snapshots}"
            )
        if len(self.snr_range) != 2:
            raise ValueError(
                f"snr_range must be a 2-tuple, got {self.snr_range}"
            )

    def build_dataset(self, array, signal_pool):
        """Build a :class:`~spectra.datasets.direction_finding.DirectionFindingDataset`.

        Args:
            array: :class:`~spectra.arrays.AntennaArray` instance.
            signal_pool: List of :class:`~spectra.waveforms.base.Waveform`.

        Returns:
            An instantiated ``DirectionFindingDataset``.
        """
        from spectra.datasets.direction_finding import DirectionFindingDataset

        return DirectionFindingDataset(
            array=array,
            signal_pool=signal_pool,
            num_signals=self.num_signals,
            num_snapshots=self.num_snapshots,
            sample_rate=self.sample_rate,
            snr_range=self.snr_range,
            num_samples=self.num_samples,
            seed=self.seed,
        )
