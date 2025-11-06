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


def linthrd2cart(s: int, q: int) -> tuple[int, int, int]:
    """Reference function for shell mapping to cartesian coordinates"""

    def get_ij(s: int, q: int) -> tuple[int, int]:
        if s == 0:
            return q, 0
        j, i = divmod(s - 1, 2 * q)
        i += j - q
        if i + j > q:
            i -= q
            j -= q + 1
        return i, j

    s_top = 2 * q * (q + 1) + 1
    if s < s_top:
        sgn = 1
        i, j = get_ij(s, q)
    else:
        sgn = -1
        i, j = get_ij(s - s_top, q - 1)

    return i, j, sgn * (q - abs(i) - abs(j))


@pytest.mark.parametrize("q", range(0, 100, 3))
def test_linthrd2cart(q):
    q_max = 4 * q**2 + 2 if q > 0 else 1
    for s in range(q_max):
        assert linthrd2cart(s, q) == asoratest.linthrd2cart(s, q)
