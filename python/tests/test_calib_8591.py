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


def test_cross_validated_fit_is_more_honest_than_in_sample():
    """The 5-fold CV r2 should be close to in-sample r2 on this well-conditioned
    synthetic problem, but slightly lower (CV cannot beat in-sample). On real
    data the gap is larger and the CV number is the honest one."""
    c = _synthetic(n=20000)
    f = cal.cross_validated_fit(c, multi=True, k=5)
    # operational coefficients match a plain fit
    f_plain = cal.fit(c, multi=True)
    assert np.allclose(f["coef_v"], f_plain["coef_v"], atol=1e-12)
    # CV r2 is at most the in-sample r2, and not wildly different on this clean
    # synthetic problem (both above 0.99).
    assert f["cv_stats_v"]["r2"] <= f["stats_v"]["r2"] + 1e-9
    assert f["cv_stats_v"]["r2"] > 0.99
    assert f["cv_k"] == 5


def test_cross_validated_fit_by_pair_holds_out_each_pair():
    """Each pair takes one turn as the held-out evaluation set, with the fit
    trained on the other pairs only."""
    rng = np.random.default_rng(0)
    pairs = {}
    for label in ("pairA", "pairB", "pairC"):
        c = _synthetic(n=5000, seed=int(rng.integers(0, 10_000)))
        pairs[label] = c
    out = cal.cross_validated_fit_by_pair(pairs, multi=True)
    assert set(out["cv_per_pair"].keys()) == {"pairA", "pairB", "pairC"}
    # each pair's held-out evaluation has the same n as its input
    for label, stats in out["cv_per_pair"].items():
        assert stats["n"] == pairs[label]["n"]
    # pooled cv stats over all held-out predictions are reasonable
    assert out["cv_stats_v"]["r2"] > 0.95
    assert out["n_held"] == sum(p["n"] for p in pairs.values())


def test_pool_dicts_complains_loudly_about_missing_numeric_key():
    import pytest
    c1 = _synthetic(n=50)
    c2 = _synthetic(n=50, seed=1)
    del c2["t85v"]                              # simulate malformed per-pair dict
    with pytest.raises(ValueError, match="t85v"):
        cal._pool_dicts([c1, c2])


def test_pool_dicts_linear_accepts_minimal_pairs():
    """Pairs without the multi-fit covariates must pool cleanly for a linear
    fit, otherwise cross_validated_fit_by_pair(..., multi=False) regresses."""
    import numpy as np
    minimal = {"n": 3, "t85v": np.arange(3.), "t85h": np.arange(3.),
               "t91v": np.arange(3.), "t91h": np.arange(3.)}
    minimal2 = {k: v + 1 for k, v in minimal.items() if k != "n"}
    minimal2["n"] = 3
    out = cal._pool_dicts([minimal, minimal2], multi=False)
    assert out["n"] == 6
    assert out["t85v"].shape == (6,)
    assert "v22" not in out


def test_pool_dicts_linear_ignores_covariates_even_if_mixed():
    """For multi=False the covariates are not part of the design, so a mixed
    presence of v22 across pairs is irrelevant and must NOT raise."""
    import numpy as np
    c1 = _synthetic(n=10)                                      # has covariates
    c2 = {"n": 10, "t85v": np.arange(10.), "t85h": np.arange(10.),
          "t91v": np.arange(10.), "t91h": np.arange(10.)}      # linear only
    out = cal._pool_dicts([c1, c2], multi=False)
    assert out["n"] == 20
    assert "v22" not in out


def test_pool_dicts_multi_requires_full_covariate_group():
    """For multi=True every per-pair dict must carry all four covariates as a
    group; a malformed set with v22 but no v37/h19/h37 must fail loudly."""
    import numpy as np, pytest
    n = 10
    half_cov = {"n": n, "t85v": np.arange(float(n)), "t85h": np.arange(float(n)),
                "t91v": np.arange(float(n)), "t91h": np.arange(float(n)),
                "v22": np.arange(float(n))}                    # v22 only
    with pytest.raises(ValueError, match="v37|h19|h37"):
        cal._pool_dicts([_synthetic(n=n), half_cov], multi=True)


def test_cross_validated_fit_by_pair_linear_path_works():
    """The public linear cross-validation path must accept linear-only pair
    dicts and return cv_stats for both V and H."""
    import numpy as np
    rng = np.random.default_rng(0)
    pairs = {}
    for label in ("A", "B", "C"):
        n = 200
        v91 = rng.uniform(230, 290, n)
        h91 = v91 - rng.uniform(0, 20, n)
        t85v = 0.97 * v91 - 2.0 + rng.normal(0, 0.3, n)
        t85h = 0.96 * h91 - 1.0 + rng.normal(0, 0.3, n)
        pairs[label] = {"n": n, "t85v": t85v, "t85h": t85h,
                        "t91v": v91, "t91h": h91}
    out = cal.cross_validated_fit_by_pair(pairs, multi=False)
    assert set(out["cv_per_pair"].keys()) == {"A", "B", "C"}
    assert out["cv_stats_v"]["r2"] > 0.9
    assert out["cv_stats_h"]["r2"] > 0.9


def test_cross_validated_fit_by_pair_validates_held_out_pair_too():
    """If a malformed pair is held out first, the order-dependent KeyError
    must be replaced with the clear ValueError from upfront validation."""
    import numpy as np, pytest
    bad = {"n": 5, "t85v": np.arange(5.), "t85h": np.arange(5.),
           "t91v": np.arange(5.), "t91h": np.arange(5.)}      # no v22 etc.
    good_a = _synthetic(n=20, seed=2)
    good_b = _synthetic(n=20, seed=3)
    with pytest.raises(ValueError, match="cross_validated_fit_by_pair.*lacks"):
        cal.cross_validated_fit_by_pair({"bad": bad, "a": good_a, "b": good_b},
                                        multi=True)


def test_validator_complains_about_missing_n_and_length_mismatch():
    """The validation helper should flag pair dicts missing 'n' or with
    arrays whose length disagrees with 'n', not let them silently slip into
    the design matrix where they would fail with raw NumPy errors."""
    import numpy as np, pytest
    base = {"n": 5, "t85v": np.arange(5.), "t85h": np.arange(5.),
            "t91v": np.arange(5.), "t91h": np.arange(5.)}
    no_n = {k: v for k, v in base.items() if k != "n"}
    with pytest.raises(ValueError, match="missing the required 'n' key"):
        cal._validate_calibration_dicts([base, no_n], multi=False)
    short = dict(base, t85v=np.arange(3.))                    # length 3 != n=5
    with pytest.raises(ValueError, match="t85v.*length 3 but n=5"):
        cal._validate_calibration_dicts([short], multi=False)
