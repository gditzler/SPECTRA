import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "verification"))
pytestmark = pytest.mark.verification


def test_lfm_properties_pass():
    import verify_lfm

    res = verify_lfm.properties()
    assert res.all_passed, res.render()


@pytest.mark.slow
def test_lfm_performance_pass():
    import verify_lfm

    res = verify_lfm.performance(full=False)
    assert res.all_passed, res.render()
