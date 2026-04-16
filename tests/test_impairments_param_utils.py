import pytest
import numpy as np
from spectra.impairments._param_utils import validate_fixed_or_random, resolve_param


class TestValidateFixedOrRandom:
    def test_both_none_raises(self):
        with pytest.raises(ValueError, match="Must provide either"):
            validate_fixed_or_random(None, None, "offset")

    def test_fixed_only_passes(self):
        validate_fixed_or_random(1.0, None, "offset")

    def test_random_only_passes(self):
        validate_fixed_or_random(None, 1.0, "offset")

    def test_both_provided_passes(self):
        validate_fixed_or_random(1.0, 2.0, "offset")


class TestResolveParam:
    def test_fixed_returns_value(self):
        assert resolve_param(5.0, None) == 5.0

    def test_random_returns_within_range(self):
        results = [resolve_param(None, 10.0) for _ in range(100)]
        assert all(-10.0 <= r <= 10.0 for r in results)

    def test_random_varies(self):
        results = [resolve_param(None, 10.0) for _ in range(20)]
        assert len(set(results)) > 1, "Random values should vary"
