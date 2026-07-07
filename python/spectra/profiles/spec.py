"""Parameter distributions and emitter profiles.

An EmitterProfile describes a real-world emitter class as a waveform type
plus sampleable physical-parameter distributions, with a citation to the
defining standard. Spec: docs/superpowers/specs/2026-07-02-waveform-realism-design.md
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Type

import numpy as np

from spectra.waveforms.base import Waveform

_MAX_DRAWS = 16


class ProfileNotRepresentable(ValueError):
    """Raised when a profile cannot fit inside the capture bandwidth."""


@dataclass(frozen=True)
class Fixed:
    value: Any

    def sample(self, rng: np.random.Generator) -> Any:
        return self.value


@dataclass(frozen=True)
class Choice:
    options: Sequence[Any]

    def __post_init__(self):
        if len(self.options) == 0:
            raise ValueError("Choice needs at least one option")

    def sample(self, rng: np.random.Generator) -> Any:
        return self.options[int(rng.integers(len(self.options)))]


@dataclass(frozen=True)
class Uniform:
    low: float
    high: float

    def __post_init__(self):
        if not self.low < self.high:
            raise ValueError(f"Uniform requires low < high, got [{self.low}, {self.high}]")

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


@dataclass(frozen=True)
class LogUniform:
    low: float
    high: float

    def __post_init__(self):
        if not 0 < self.low < self.high:
            raise ValueError(f"LogUniform requires 0 < low < high, got [{self.low}, {self.high}]")

    def sample(self, rng: np.random.Generator) -> float:
        return float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))


@dataclass(frozen=True)
class EmitterProfile:
    """A standards-referenced emitter: waveform class + parameter distributions.

    Attributes:
        name: Registry key, kebab-case (e.g. ``"bluetooth-le-1m"``).
        label: Dataset class label (e.g. ``"BLE"``); becomes
            ``SignalDescription.label`` in scenes.
        waveform_cls: Waveform class constructed by :meth:`sample`.
        params: Mapping of constructor kwarg -> ParamSpec.
        reference: One-line citation of the defining standard.
    """

    name: str
    label: str
    waveform_cls: Type[Waveform]
    params: Mapping[str, Any] = field(default_factory=dict)
    reference: str = ""

    def sample(self, rng: np.random.Generator, sample_rate: float) -> Waveform:
        """Draw a concrete waveform representable at ``sample_rate``.

        Re-draws up to a fixed budget when a draw's occupied bandwidth
        exceeds ``sample_rate``; profiles never silently distort parameters.
        """
        last_bw = None
        for _ in range(_MAX_DRAWS):
            kwargs = {k: spec.sample(rng) for k, spec in self.params.items()}
            wf = self.waveform_cls(**kwargs)
            last_bw = wf.bandwidth(sample_rate)
            # bandwidth() alone is not sufficient: symbol-rate-parameterized
            # waveforms also need >= 2 samples/symbol at generate() time.
            sr = getattr(wf, "symbol_rate", None)
            if last_bw <= sample_rate and (sr is None or 2 * sr <= sample_rate):
                return wf
        raise ProfileNotRepresentable(
            f"profile '{self.name}' drew bandwidth {last_bw:g} Hz > sample_rate "
            f"{sample_rate:g} Hz after {_MAX_DRAWS} attempts; increase the capture "
            f"sample rate or remove this profile from the pool"
        )
