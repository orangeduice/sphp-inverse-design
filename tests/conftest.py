import os
from pathlib import Path

import pytest

# The dataset lives outside git. Override with SPHP_DATA=/path/to/2026_04_15_full_pisa_data.h5
DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data" / "2026_04_15_full_pisa_data.h5"


@pytest.fixture(scope="session")
def data_path():
    p = Path(os.environ.get("SPHP_DATA", DEFAULT_DATA))
    if not p.exists():
        pytest.skip(f"dataset not found at {p} (set SPHP_DATA)")
    return p
