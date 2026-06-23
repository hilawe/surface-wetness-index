"""Regression: the NumPy port must reproduce the C oracle on every cell.

The C engine (core_c) is the bit-for-bit reference. These tests run a large,
branch-covering sample through both engines and require exact agreement on the
integer outputs (SNOW, return code) and the float32 outputs (RTEMP, WET).
"""

import numpy as np
import pytest

from swi import core_c, core_numpy
from swi.channels import kelvin_to_packed
from scripts.regression_check import make_sample


def _float_equal(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return (a == b) | (np.isnan(a) & np.isnan(b))


def _compare(chan):
    c = core_c.evaluate_packed(chan)
    p = core_numpy.evaluate_packed(chan)
    np.testing.assert_array_equal(c.snow, p.snow)
    np.testing.assert_array_equal(c.ret, p.ret)
    assert _float_equal(c.temp, p.temp).all()
    assert _float_equal(c.wet, p.wet).all()
    return c


@pytest.mark.parametrize("seed", [1, 2, 7])
def test_random_sample_matches_oracle(seed):
    with np.errstate(divide="ignore", invalid="ignore"):
        chan = make_sample(1_000_000, seed)
        c = _compare(chan)
    # Sanity: the sample actually exercises the deep branches.
    assert (c.wet > 0).sum() > 100        # wet-surface detections
    assert (c.snow > 0).sum() > 1000      # snow scattering
    assert set(np.unique(c.ret)) == {-1, 0, 1}


def test_full_byte_domain_smoke():
    """A structured small grid plus the gap/fill sentinel."""
    chan = np.array([
        [32, 32, 32, 32, 32, 32, 32],          # fill -> gap
        [200, 185, 202, 198, 185, 195, 188],   # warm vegetated-ish
        [150, 148, 160, 175, 170, 185, 178],   # cold scattering
    ], dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        _compare(chan)


def test_kelvin_roundtrip_matches_packed():
    rng = np.random.default_rng(0)
    tb = rng.uniform(150, 300, size=(5000, 7))
    with np.errstate(divide="ignore", invalid="ignore"):
        a = core_numpy.evaluate_kelvin(tb)
        b = core_numpy.evaluate_packed(kelvin_to_packed(tb))
    np.testing.assert_array_equal(a.snow, b.snow)
    assert _float_equal(a.wet, b.wet).all()
