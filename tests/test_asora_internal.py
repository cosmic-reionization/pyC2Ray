import numpy as np
import pytest

try:
    from pyc2ray.lib import libasoratest as asoratest
except ImportError:
    pytest.skip("libasoratest.so missing, skipping tests", allow_module_level=True)


@pytest.mark.parametrize("pos0", [(5, 5, 5), (1, 2, 3), (10, 5, 15)])
def test_cinterp(pos0):
    rng = np.random.default_rng(seed=42)
    N = 20
    dens = rng.random((N, N, N), dtype=np.float64)

    output = asoratest.cinterp(pos0, dens)
    suff = "".join(f"{s:02}" for s in pos0)
    expected_output = np.load(f"tests/data/cinterp_output_{suff}.npy")

    assert np.allclose(output, expected_output, equal_nan=True)
