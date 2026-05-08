import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "verification"))
pytestmark = pytest.mark.verification


def test_adsb_properties_pass():
    import verify_adsb

    res = verify_adsb.properties()
    assert res.all_passed, res.render()
