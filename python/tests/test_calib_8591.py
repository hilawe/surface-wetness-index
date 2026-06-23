"""85-to-91 calibration math: recover a known relationship and apply it."""

import numpy as np

from swi import calib_8591 as cal
from swi.channels import (CH19V, CH19H, CH22V, CH37V, CH37H, CH85V, CH85H,
                          N_CHANNELS)


def _synthetic(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    v91 = rng.uniform(230, 290, n)
    h91 = v91 - rng.uniform(0, 20, n)
    v22 = rng.uniform(240, 290, n)
    v37 = rng.uniform(235, 290, n)
    h37 = v37 - rng.uniform(0, 15, n)
    h19 = rng.uniform(220, 285, n)
    # known truth: 85V = 0.95*91V + 0.04*22V + 0.02*37V - 3 ; 85H = 0.97*91H + 2
    t85v = 0.95 * v91 + 0.04 * v22 + 0.02 * v37 - 3.0 + rng.normal(0, 0.2, n)
    t85h = 0.97 * h91 + 0.03 * h19 - 1.0 + rng.normal(0, 0.2, n)
    return {"n": n, "t85v": t85v, "t85h": t85h, "t91v": v91, "t91h": h91,
            "v22": v22, "v37": v37, "h19": h19, "h37": h37}


def test_fit_recovers_and_is_accurate():
    c = _synthetic()
    f = cal.fit(c, multi=True)
    # 91V coefficient near the true 0.95
    assert abs(f["coef_v"][0] - 0.95) < 0.02
    # tight fit
    assert f["stats_v"]["rms"] < 0.5 and f["stats_v"]["r2"] > 0.99
    assert f["stats_h"]["rms"] < 0.5 and f["stats_h"]["r2"] > 0.99
    # the naive 91-85 bias is reported and nonzero
    assert abs(f["raw_bias_v"]) > 0.1


def test_apply_matches_fit_prediction():
    c = _synthetic()
    f = cal.fit(c, multi=True)
    n = c["n"]
    cube = np.full((n, N_CHANNELS), np.nan, dtype=np.float32)
    cube[:, CH19V] = 270
    cube[:, CH19H] = 255
    cube[:, CH22V] = c["v22"]
    cube[:, CH37V] = c["v37"]
    cube[:, CH37H] = c["h37"]
    cube[:, CH85V] = c["t91v"]      # 91 sits in the 85 slot
    cube[:, CH85H] = c["t91h"]
    out = cal.apply(cube, f)
    # the 85V slot now holds the fitted prediction
    bv = f["coef_v"]
    expected = bv[0] * c["t91v"] + bv[1] * c["v22"] + bv[2] * c["v37"] + bv[3]
    assert np.allclose(out[:, CH85V], expected.astype(np.float32), atol=1e-3)
    # untouched channels preserved
    assert np.allclose(out[:, CH22V], c["v22"].astype(np.float32))


def test_apply_preserves_nan():
    c = _synthetic(n=100)
    f = cal.fit(c)
    cube = np.full((3, N_CHANNELS), np.nan, dtype=np.float32)
    out = cal.apply(cube, f)
    assert np.isnan(out).all()
