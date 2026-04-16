"""Shared parameter validation and resolution for impairment transforms."""

from typing import Optional

import numpy as np


def validate_fixed_or_random(
    fixed: Optional[float],
    max_val: Optional[float],
    name: str,
) -> None:
    """Validate that at least one of fixed or max_val is provided."""
    if fixed is None and max_val is None:
        raise ValueError(f"Must provide either {name} or max_{name}")


def resolve_param(fixed: Optional[float], max_val: Optional[float]) -> float:
    """Return the fixed value or sample uniformly from [-max_val, max_val]."""
    if max_val is not None:
        return float(np.random.uniform(-max_val, max_val))
    return float(fixed)
