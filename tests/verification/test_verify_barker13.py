import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "verification"))
pytestmark = pytest.mark.verification


def test_barker13_properties_pass():
    import verify_barker13

    res = verify_barker13.properties()
    assert res.all_passed, res.render()


@pytest.mark.slow
def test_barker13_performance_pass():
    import verify_barker13

    res = verify_barker13.performance(full=False)
    assert res.all_passed, res.render()
