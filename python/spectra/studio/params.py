# python/spectra/studio/params.py
"""Waveform parameter registry for SPECTRA Studio.

Auto-discovers constructor parameters from waveform classes via
``inspect.signature()``, providing the UI with type, default, and
label information for dynamic parameter rendering.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List

from spectra.cli.config_builder import WAVEFORM_CATEGORIES, get_waveform_registry

# Parameters to exclude from UI (internal/inherited)
_EXCLUDE_PARAMS = {"self", "args", "kwargs"}

# Choice-type overrides for specific parameters
_CHOICE_OVERRIDES: Dict[str, List[str]] = {
    "pulse_shape": ["rect", "hamming", "hann"],
    "code_type": ["frank", "p1", "p2", "p3", "p4"],
    "sweep_type": ["sawtooth", "triangle", "tandem_hooked", "s_curve"],
    "prf_mode": ["low", "medium", "high"],
}


def get_waveform_params(waveform_name: str) -> List[Dict[str, Any]]:
    """Get UI-renderable parameter descriptors for a waveform class.

    Returns a list of dicts, each with keys:
    ``name``, ``type`` (``"int"``, ``"float"``, ``"choice"``),
    ``default``, ``label``, and optionally ``choices``.

    Returns empty list for unknown waveform names.
    """
    registry = get_waveform_registry()
    cls = registry.get(waveform_name)
    if cls is None:
        return []

    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return []

    params = []
    for name, param in sig.parameters.items():
        if name in _EXCLUDE_PARAMS:
            continue

        default = param.default if param.default is not inspect.Parameter.empty else None
        annotation = param.annotation

        # Determine UI type
        if name in _CHOICE_OVERRIDES:
            p_type = "choice"
            choices = _CHOICE_OVERRIDES[name]
        elif annotation is int or isinstance(default, int):
            p_type = "int"
            choices = None
        elif annotation is float or isinstance(default, (float, int)):
            p_type = "float"
            choices = None
        elif isinstance(default, str):
            p_type = "choice"
            choices = _CHOICE_OVERRIDES.get(name, [str(default)])
        else:
            p_type = "float"
            choices = None

        label = name.replace("_", " ").title()

        entry: Dict[str, Any] = {
            "name": name,
            "type": p_type,
            "default": default,
            "label": label,
        }
        if choices:
            entry["choices"] = choices

        params.append(entry)

    return params


def get_all_categories() -> List[str]:
    """Return all waveform category names."""
    return list(WAVEFORM_CATEGORIES.keys())


def get_waveforms_for_category(category: str) -> List[str]:
    """Return waveform names for a given category."""
    return WAVEFORM_CATEGORIES.get(category, [])
