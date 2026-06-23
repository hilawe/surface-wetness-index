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


def fit(c, multi=True):
    """Fit the 85-from-91 adjustment. Returns coefficients and diagnostics."""
    n = c["n"]
    ones = np.ones(n)
    if multi:
        Xv = np.column_stack([c["t91v"], c["v22"], c["v37"], ones])
        Xh = np.column_stack([c["t91h"], c["h19"], c["h37"], ones])
    else:
        Xv = np.column_stack([c["t91v"], ones])
        Xh = np.column_stack([c["t91h"], ones])
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
        "n": n,
    }


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
