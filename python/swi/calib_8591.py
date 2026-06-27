"""85-to-91 GHz calibration for feeding SSMIS into the SSM/I-tuned algorithm.

Basist's coefficients were tuned on 85.5 GHz (SSM/I). SSMIS measures 91.665 GHz.
The wetness retrieval is acutely sensitive to the 85 GHz channel (a ~1 K bias
swings wet-surface detections by ~25 percent), so feeding raw 91 GHz into the
85 GHz slots biases the product. This module derives, from overlap-era files
where an SSM/I sensor (true 85) and an SSMIS sensor (91) flew at the same time,
an adjustment that predicts the SSM/I 85 GHz a SSMIS observation would correspond
to, then applies it.

Method (first-order, gridded): on the common 0.25 degree grid, over land, match
cells and fit
    85V_SSMI ~ a + b*91V_SSMIS [+ c*22V + d*37V]
    85H_SSMI ~ a'+ b'*91H_SSMIS[+ c'*19H + d'*37H]
The common channels (19/22/37 GHz, measured by both sensors) are used as
covariates to absorb part of the time-of-day and scene difference between the two
overpasses. At apply time only SSMIS channels are needed.

Caveats (documented for the CDR record):
- The two sensors observe at different local times, so the fit absorbs a residual
  diurnal/temperature difference into the intercept. Prefer SSM/I sensors with a
  crossing time close to the SSMIS one, and avoid F-15 (RADCAL 85 GHz
  interference from 2006). Pool several dates across seasons.
- A footprint-level swath collocation is the rigorous version if this first-order
  gridded adjustment proves insufficient.
"""

import numpy as np

from . import io_csu_grid as io
from .channels import (CH19V, CH19H, CH22V, CH37V, CH37H, CH85V, CH85H,
                       N_CHANNELS)


def land_mask(lat, lon, tb=None):
    """Boolean (nlat, nlon) land mask.

    Uses global_land_mask if available; otherwise a polarization-based proxy
    (ocean has large 19/37 GHz polarization differences).
    """
    try:
        from global_land_mask import globe
        lon2 = np.where(lon > 180, lon - 360, lon)            # globe wants -180..180
        LON, LAT = np.meshgrid(lon2, lat)
        return globe.is_land(LAT, LON)
    except Exception:
        if tb is None:
            raise
        pd19 = tb[..., CH19V] - tb[..., CH19H]
        pd37 = tb[..., CH37V] - tb[..., CH37H]
        return (pd19 < 35) & (pd37 < 25)


def collocate(ssmi_file, ssmis_file, pass_="dsc", land=True):
    """Match an SSM/I file (true 85) with an SSMIS file (91) on the common grid.

    Returns a dict of 1-D arrays at matched (land) cells: the SSM/I 85V/85H
    targets and the SSMIS 91V/91H plus common channels as predictors.
    """
    la1, lo1, tb1, s1 = io.read_channels(ssmi_file, pass_=pass_)    # SSM/I, true 85
    la2, lo2, tb2, s2 = io.read_channels(ssmis_file, pass_=pass_)   # SSMIS, 91 in 85 slots
    if s1 != "ssmi":
        raise ValueError(f"{ssmi_file}: expected an SSM/I file (true 85 GHz), got {s1}")
    if s2 != "ssmis":
        raise ValueError(f"{ssmis_file}: expected an SSMIS file (91 GHz), got {s2}")
    if la1.shape != la2.shape or lo1.shape != lo2.shape:
        raise ValueError("files are on different grids")

    m = np.isfinite(tb1).all(-1) & np.isfinite(tb2).all(-1)
    if land:
        m &= land_mask(la1, lo1, tb1)
    return {
        "n": int(m.sum()),
        "t85v": tb1[m, CH85V], "t85h": tb1[m, CH85H],          # targets (SSM/I 85)
        "t91v": tb2[m, CH85V], "t91h": tb2[m, CH85H],          # SSMIS 91 (in 85 slot)
        "v22": tb2[m, CH22V], "v37": tb2[m, CH37V],
        "h19": tb2[m, CH19H], "h37": tb2[m, CH37H],
    }


def pool(pairs, pass_="dsc", land=True):
    """Collocate and concatenate several (ssmi_file, ssmis_file) date pairs."""
    parts = [collocate(a, b, pass_=pass_, land=land) for a, b in pairs]
    keys = [k for k in parts[0] if k != "n"]
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out["n"] = sum(p["n"] for p in parts)
    return out


def _stats(pred, true):
    res = true - pred
    return {"rms": float(np.sqrt(np.mean(res ** 2))),
            "bias": float(res.mean()),
            "r2": float(1.0 - np.var(res) / np.var(true))}


def _design_matrices(c, multi):
    ones = np.ones(c["n"])
    if multi:
        Xv = np.column_stack([c["t91v"], c["v22"], c["v37"], ones])
        Xh = np.column_stack([c["t91h"], c["h19"], c["h37"], ones])
    else:
        Xv = np.column_stack([c["t91v"], ones])
        Xh = np.column_stack([c["t91h"], ones])
    return Xv, Xh


def fit(c, multi=True):
    """Fit the 85-from-91 adjustment. Returns coefficients and diagnostics.

    Diagnostics are in-sample by default. Use cross_validated_fit() to get a
    held-out estimate of the fit quality on data not used to determine the
    coefficients.
    """
    Xv, Xh = _design_matrices(c, multi)
    bv, *_ = np.linalg.lstsq(Xv, c["t85v"], rcond=None)
    bh, *_ = np.linalg.lstsq(Xh, c["t85h"], rcond=None)
    return {
        "model": "multi" if multi else "linear",
        "coef_v": bv, "coef_h": bh,
        "stats_v": _stats(Xv @ bv, c["t85v"]),
        "stats_h": _stats(Xh @ bh, c["t85h"]),
        # naive characterization for reference
        "raw_bias_v": float((c["t91v"] - c["t85v"]).mean()),
        "raw_bias_h": float((c["t91h"] - c["t85h"]).mean()),
        "n": c["n"],
    }


def cross_validated_fit(c, multi=True, k=5, seed=0):
    """k-fold cross-validated fit quality for the 85-from-91 adjustment.

    Splits the pooled cells into k random folds, fits on k-1 folds, scores on the
    held-out fold, and aggregates the held-out RMS, bias, and r2 across all
    folds. The full-data coefficients are also returned so the saved fit uses
    the same model that would have been picked without cross-validation; CV
    diagnostics are the honest estimate of how that model performs on data
    outside the fit. Default k=5.

    This addresses the adversarial-review finding that the 85-to-91 calibration
    reports in-sample fit diagnostics only. Use the cv_stats fields as the
    headline; the in-sample stats are inflated by definition.
    """
    n = c["n"]
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, k, size=n)
    Xv, Xh = _design_matrices(c, multi)
    yv = c["t85v"]
    yh = c["t85h"]
    preds_v = np.empty(n)
    preds_h = np.empty(n)
    for f in range(k):
        train = fold != f
        test = fold == f
        bv, *_ = np.linalg.lstsq(Xv[train], yv[train], rcond=None)
        bh, *_ = np.linalg.lstsq(Xh[train], yh[train], rcond=None)
        preds_v[test] = Xv[test] @ bv
        preds_h[test] = Xh[test] @ bh
    # Full-data coefficients for the operational fit.
    bv_full, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    bh_full, *_ = np.linalg.lstsq(Xh, yh, rcond=None)
    return {
        "model": "multi" if multi else "linear",
        "coef_v": bv_full, "coef_h": bh_full,
        "stats_v": _stats(Xv @ bv_full, yv),
        "stats_h": _stats(Xh @ bh_full, yh),
        "cv_stats_v": _stats(preds_v, yv),
        "cv_stats_h": _stats(preds_h, yh),
        "cv_k": k,
        "cv_seed": seed,
        "raw_bias_v": float((c["t91v"] - yv).mean()),
        "raw_bias_h": float((c["t91h"] - yh).mean()),
        "n": n,
    }


def cross_validated_fit_by_pair(c_by_pair, multi=True):
    """Leave-one-pair-out cross-validation across satellite or date pairs.

    Per-satellite-pair holdout is a stronger test than random k-fold because the
    leakage we worry about is shared sensor characteristics within one pair, not
    random per-cell noise. c_by_pair is a dict mapping pair-label to a per-pair
    pooled cell dict (as returned by collocate or pool on a single pair). The
    function trains on every-pair-except-one and evaluates on the held-out
    pair, repeated for each pair. Returns a dict with cv_per_pair and pooled
    cv_stats_v/cv_stats_h on the union of held-out predictions.

    For the operational coefficients, use fit() on pool(...) of all pairs.
    """
    labels = list(c_by_pair.keys())
    if len(labels) < 2:
        raise ValueError("need at least two pairs for leave-one-pair-out CV")
    # Validate every pair up front, so a malformed held-out pair fails with the
    # clear ValueError instead of a raw KeyError later inside _design_matrices.
    _validate_calibration_dicts([c_by_pair[l] for l in labels], multi,
                                where="cross_validated_fit_by_pair")
    pair_stats = {}
    pv_all, yv_all, ph_all, yh_all = [], [], [], []
    for held in labels:
        train_labels = [l for l in labels if l != held]
        c_train = _pool_dicts([c_by_pair[l] for l in train_labels], multi=multi)
        Xv_tr, Xh_tr = _design_matrices(c_train, multi)
        bv, *_ = np.linalg.lstsq(Xv_tr, c_train["t85v"], rcond=None)
        bh, *_ = np.linalg.lstsq(Xh_tr, c_train["t85h"], rcond=None)
        c_held = c_by_pair[held]
        Xv_he, Xh_he = _design_matrices(c_held, multi)
        pv = Xv_he @ bv
        ph = Xh_he @ bh
        pair_stats[held] = {
            "stats_v": _stats(pv, c_held["t85v"]),
            "stats_h": _stats(ph, c_held["t85h"]),
            "n": c_held["n"],
        }
        pv_all.append(pv); yv_all.append(c_held["t85v"])
        ph_all.append(ph); yh_all.append(c_held["t85h"])
    pv_all = np.concatenate(pv_all)
    ph_all = np.concatenate(ph_all)
    yv_all = np.concatenate(yv_all)
    yh_all = np.concatenate(yh_all)
    return {
        "cv_per_pair": pair_stats,
        "cv_stats_v": _stats(pv_all, yv_all),
        "cv_stats_h": _stats(ph_all, yh_all),
        "n_held": pv_all.size,
    }


# Core targets and predictors needed by every fit shape. Linear fits use only
# these four (true 85 channels and the 91 predictors); multi fits add the
# covariates below for the V and H designs.
_CORE_KEYS = ("t85v", "t85h", "t91v", "t91h")
# Covariates added by multi fits.
_COVARIATE_KEYS = ("v22", "v37", "h19", "h37")


def _validate_calibration_dicts(cs, multi, where="_pool_dicts"):
    """Raise ValueError if any per-pair dict in cs is missing keys required for
    the requested fit shape, lacks 'n', or has arrays whose lengths do not
    agree with 'n'. Centralised so the error path is the same whether we are
    pooling for training or evaluating a single held-out pair.
    """
    for i, c in enumerate(cs):
        if "n" not in c:
            raise ValueError(
                f"{where}: per-pair dict {i} is missing the required 'n' key")
    for k in _CORE_KEYS:
        for i, c in enumerate(cs):
            if k not in c:
                raise ValueError(
                    f"{where}: required calibration key {k!r} missing from "
                    f"per-pair dict {i}; cannot fit")
    if multi:
        missing_in_pair = []
        for i, c in enumerate(cs):
            absent = [k for k in _COVARIATE_KEYS if k not in c]
            if absent:
                missing_in_pair.append((i, absent))
        if missing_in_pair:
            raise ValueError(
                f"{where}(multi=True) needs every per-pair dict to carry all "
                f"of {_COVARIATE_KEYS}. Missing in pair(s): "
                + ", ".join(f"{i} (lacks {ks})" for i, ks in missing_in_pair)
                + ". Use multi=False for a linear pool across these pairs.")
    # Length agreement: every numeric array key present must match the pair's n.
    arr_keys = list(_CORE_KEYS) + (list(_COVARIATE_KEYS) if multi else [])
    for i, c in enumerate(cs):
        n = c["n"]
        for k in arr_keys:
            if k in c and len(c[k]) != n:
                raise ValueError(
                    f"{where}: per-pair dict {i} has key {k!r} of length "
                    f"{len(c[k])} but n={n}; arrays must match n")


def _pool_dicts(cs, multi=True):
    """Concatenate per-pair pooled cell dicts on the numeric calibration keys.

    The four core keys (true 85 channels and the 91 predictors) must be present
    in every per-pair dict; a missing one is a real input error and raises a
    ValueError naming the key and the pair index. When multi=True, every pair
    must also carry all four covariates as a complete group: missing any one
    in any pair is an error, so the caller cannot silently slip into a
    linear-shaped pool when they asked for a multi fit. When multi=False, the
    covariates are simply ignored, and pairs may freely carry or not carry
    them; this is the path used by linear cross-validation when pair dicts
    were cached for an earlier multi-shape run.
    Metadata keys (paths, labels) on per-pair dicts are ignored either way.
    """
    _validate_calibration_dicts(cs, multi, where="_pool_dicts")
    out = {k: np.concatenate([c[k] for c in cs]) for k in _CORE_KEYS}
    if multi:
        for k in _COVARIATE_KEYS:
            out[k] = np.concatenate([c[k] for c in cs])
    out["n"] = sum(c["n"] for c in cs)
    return out


def save_fit(coeffs, path, meta=None):
    """Persist a fit as JSON (coefficients, diagnostics, and provenance)."""
    import json
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
           for k, v in coeffs.items()}
    out["meta"] = meta or {}
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    return path


def load_fit(path):
    """Load a fit saved by save_fit; restores coef arrays."""
    import json
    with open(path) as fh:
        d = json.load(fh)
    d["coef_v"] = np.asarray(d["coef_v"], dtype=np.float64)
    d["coef_h"] = np.asarray(d["coef_h"], dtype=np.float64)
    return d


def apply(tb_cube, coeffs):
    """Return an SSMIS tb cube with 85-equivalent values in the 85 (P6/P7) slots.

    NaNs propagate. Input may be (..., 7); the 19/22/37 GHz slots are read for the
    multi-channel model and left unchanged.
    """
    out = np.array(tb_cube, dtype=np.float64, copy=True)
    if out.shape[-1] != N_CHANNELS:
        raise ValueError(f"last axis must be {N_CHANNELS}")
    v91 = out[..., CH85V].copy()
    h91 = out[..., CH85H].copy()
    bv, bh = coeffs["coef_v"], coeffs["coef_h"]
    if coeffs["model"] == "multi":
        out[..., CH85V] = bv[0] * v91 + bv[1] * out[..., CH22V] + bv[2] * out[..., CH37V] + bv[3]
        out[..., CH85H] = bh[0] * h91 + bh[1] * out[..., CH19H] + bh[2] * out[..., CH37H] + bh[3]
    else:
        out[..., CH85V] = bv[0] * v91 + bv[1]
        out[..., CH85H] = bh[0] * h91 + bh[1]
    return out.astype(np.float32)



def make_engine(coeffs):
    """Return an engine that applies the 85-to-91 adjustment then evaluates.

    The returned object has an `evaluate_kelvin(tb)` method matching the
    interface that `io.evaluate_file` expects. With `coeffs=None` it is a
    passthrough to `core_numpy.evaluate_kelvin`. Centralizes a small wrapper
    that used to be copy-pasted across four product-writer scripts.
    """
    from . import core_numpy

    class _CalibEngine:
        def __init__(self, coeffs):
            self.coeffs = coeffs

        def evaluate_kelvin(self, tb):
            if self.coeffs is not None:
                tb = apply(tb, self.coeffs)
            return core_numpy.evaluate_kelvin(tb)

    return _CalibEngine(coeffs)
