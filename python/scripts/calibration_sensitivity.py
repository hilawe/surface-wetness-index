"""How far do the decision tree's outputs move under the 85-to-91 GHz calibration error?

The SSMIS-era product needs an empirical mapping from the 91.655 GHz channels to the
85.5 GHz the tree expects. That mapping is not exact, with residuals of about
2.85 K at vertical polarization and 3.23 K at horizontal, while
`the project documentation` argues the tree is sensitive at the roughly 1 K level.
Reporting the mapping's mean bias alone hides that, because a small mean bias is
consistent with large per-cell error.

This driver propagates the error instead of quoting it. It takes a day of SSMIS input,
applies the saved calibration, and then perturbs the resulting 85 GHz channels by draws
from the residual distribution, rerunning the whole tree each time. What it reports is
how often the tree's decisions change, which is the quantity that matters to a user of
the product.

Two perturbation modes, because they answer different questions:

  random  independent per-cell noise at the residual scale. This is a stress
          case, and it asks how reproducible a cell's classification is.
  bias    a uniform offset applied everywhere. This asks how far a systematic
          calibration error would move the global statistics, and it tests the
          claim in the calibration memo that 1 K matters.

A third mode, --paired, redraws actual (V, H) residual pairs from the fit's own
collocated cells instead of independent Gaussian noise. The vertical and
horizontal residuals are correlated, so paired draws are the more realistic
per-cell error model, and the difference between the paired and independent flip
fractions bounds the error-model dependence. It needs the overlap files named in
the calibration JSON's metadata to be on disk.

Usage:
    python -m scripts.calibration_sensitivity CSU_DAILY.nc --calib CAL.json
        [--pass dsc] [--draws 25] [--paired] [--out J.json]
"""

import json
import os
import sys

import numpy as np

from swi import calib_8591, core_numpy, grids, io_csu_grid

WET_SENTINELS = -90.0          # below this the cell is a sentinel, not a retrieval


def opt(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def _outputs(tb, coeffs, dv=0.0, dh=0.0, rng=None, sd_v=0.0, sd_h=0.0):
    """Run the tree on calibrated Tb, optionally perturbing the 85 GHz channels."""
    t = tb.copy()
    if coeffs is not None:
        t = calib_8591.apply(t, coeffs)
    if rng is not None:
        t[:, :, 5] += rng.normal(0.0, sd_v, t.shape[:2])
        t[:, :, 6] += rng.normal(0.0, sd_h, t.shape[:2])
    else:
        t[:, :, 5] += dv
        t[:, :, 6] += dh
    return core_numpy.evaluate_kelvin(t.reshape(-1, 7))


def main():
    argv = sys.argv
    if len(argv) < 2:
        print(__doc__)
        return 1
    path = argv[1]
    calib = opt(argv, "--calib", "../calibration/f15_f16_2006_dsc_v0.json")
    pass_ = opt(argv, "--pass", "dsc")
    draws = int(opt(argv, "--draws", "25"))
    out_path = opt(argv, "--out", "../results/calibration_sensitivity.json")

    coeffs = calib_8591.load_fit(calib)
    sd_v = float(coeffs["stats_v"]["rms"])
    sd_h = float(coeffs["stats_h"]["rms"])

    lat, lon, tb, sensor = io_csu_grid.read_channels(path, pass_=pass_)
    land = grids.land_mask(lat, lon).reshape(-1)

    base = _outputs(tb, coeffs)
    b_wet = np.asarray(base.wet, np.float64).reshape(-1)
    b_snow = np.asarray(base.snow, np.float64).reshape(-1)
    b_tmp = np.asarray(base.temp, np.float64).reshape(-1)
    valid = land & (b_wet > WET_SENTINELS)
    b_fire = b_wet > 0.0

    res = {"file": os.path.basename(path), "sensor": sensor, "pass": pass_,
           "calibration": os.path.basename(calib),
           "residual_rms_k": {"85v": sd_v, "85h": sd_h},
           "n_valid_land_cells": int(valid.sum()),
           "baseline_firing_fraction": float(b_fire[valid].mean())}

    print(f"\n{os.path.basename(path)}  sensor={sensor} pass={pass_}")
    print(f"calibration residual RMS: 85V {sd_v:.2f} K, 85H {sd_h:.2f} K")
    print(f"valid land cells {int(valid.sum()):,}, "
          f"baseline firing fraction {float(b_fire[valid].mean()):.4f}\n")

    # 1. Realistic per-cell noise at the residual scale.
    rng = np.random.RandomState(20260729)
    flip, dfire, dwet, dsnow, dropped = [], [], [], [], []
    for _ in range(draws):
        o = _outputs(tb, coeffs, rng=rng, sd_v=sd_v, sd_h=sd_h)
        w = np.asarray(o.wet, np.float64).reshape(-1)
        s = np.asarray(o.snow, np.float64).reshape(-1)
        v = valid & (w > WET_SENTINELS)
        dropped.append(float((valid & ~(w > WET_SENTINELS)).sum() / valid.sum()))
        flip.append(float((( w > 0.0) != b_fire)[v].mean()))
        dfire.append(float((w > 0.0)[v].mean() - b_fire[v].mean()))
        dwet.append(float(np.abs(w[v] - b_wet[v]).mean()))
        dsnow.append(float((s[v] != b_snow[v]).mean()))
    res["random_perturbation"] = {
        "draws": draws, "seed": 20260729,
        "invalidated_frac_mean": float(np.mean(dropped)),
        "denominator": ("flip fractions are over baseline-valid cells that "
                        "remain valid under the perturbation; the invalidated "
                        "fraction is reported separately"),
        "wet_classification_flip_frac_mean": float(np.mean(flip)),
        "wet_classification_flip_frac_max": float(np.max(flip)),
        "firing_fraction_shift_mean": float(np.mean(dfire)),
        "mean_abs_wet_change": float(np.mean(dwet)),
        "snow_flag_change_frac_mean": float(np.mean(dsnow)),
    }
    print("Per-cell noise at the residual scale "
          f"({draws} draws):")
    print(f"  wet/dry classification flips : {100*np.mean(flip):.2f}% of cells "
          f"(worst draw {100*np.max(flip):.2f}%)")
    print(f"  mean |change| in WET index   : {np.mean(dwet):.2f} on the 0 to 100 scale")
    print(f"  snow flag changes            : {100*np.mean(dsnow):.2f}% of cells")

    # 2. Paired empirical residuals: redraw actual (V, H) residual pairs from
    # the fit's collocated cells, preserving the V-H residual correlation.
    if "--paired" in argv:
        with open(calib) as fh:
            raw = json.load(fh)
        pairs = [tuple(p) for p in raw["meta"]["pairs"]]
        multi = raw.get("model", raw["meta"].get("model", "multi")) == "multi"
        print(f"\ncollocating {len(pairs)} overlap pair(s) for empirical residuals ...")
        c = calib_8591.pool(pairs, pass_=raw["meta"]["pass"])
        Xv, Xh = calib_8591._design_matrices(c, multi)
        rv = c["t85v"] - Xv @ np.asarray(coeffs["coef_v"], np.float64)
        rh = c["t85h"] - Xh @ np.asarray(coeffs["coef_h"], np.float64)
        corr_vh = float(np.corrcoef(rv, rh)[0, 1])
        rng_p = np.random.RandomState(20260730)
        ncell = tb.shape[0] * tb.shape[1]
        p_flip, p_dfire, p_dwet, p_dsnow, p_dropped = [], [], [], [], []
        for _ in range(draws):
            idx = rng_p.randint(0, rv.size, ncell)
            t = calib_8591.apply(tb.copy(), coeffs)
            t[:, :, 5] += rv[idx].reshape(tb.shape[:2])
            t[:, :, 6] += rh[idx].reshape(tb.shape[:2])
            o = core_numpy.evaluate_kelvin(t.reshape(-1, 7))
            w = np.asarray(o.wet, np.float64).reshape(-1)
            s = np.asarray(o.snow, np.float64).reshape(-1)
            v = valid & (w > WET_SENTINELS)
            p_dropped.append(float((valid & ~(w > WET_SENTINELS)).sum() / valid.sum()))
            p_flip.append(float(((w > 0.0) != b_fire)[v].mean()))
            p_dfire.append(float((w > 0.0)[v].mean() - b_fire[v].mean()))
            p_dwet.append(float(np.abs(w[v] - b_wet[v]).mean()))
            p_dsnow.append(float((s[v] != b_snow[v]).mean()))
        res["paired_residual_perturbation"] = {
            "draws": draws, "seed": 20260730,
            "invalidated_frac_mean": float(np.mean(p_dropped)),
            "n_residual_pairs": int(rv.size),
            "residual_vh_correlation": corr_vh,
            "wet_classification_flip_frac_mean": float(np.mean(p_flip)),
            "wet_classification_flip_frac_max": float(np.max(p_flip)),
            "firing_fraction_shift_mean": float(np.mean(p_dfire)),
            "mean_abs_wet_change": float(np.mean(p_dwet)),
            "snow_flag_change_frac_mean": float(np.mean(p_dsnow)),
        }
        print(f"Paired empirical residuals ({draws} draws, "
              f"{rv.size:,} residual pairs, V-H correlation {corr_vh:+.2f}):")
        print(f"  wet/dry classification flips : {100*np.mean(p_flip):.2f}% of cells "
              f"(worst draw {100*np.max(p_flip):.2f}%)")
        print(f"  firing-fraction shift (mean) : {np.mean(p_dfire):+.4f}")
        print(f"  mean |change| in WET index   : {np.mean(p_dwet):.2f}")
        print(f"  snow flag changes            : {100*np.mean(p_dsnow):.2f}% of cells")

    # 3. Systematic offsets, including the 1 K case the calibration memo flags.
    res["bias_perturbation"] = {}
    print("\nUniform offset applied to both 85 GHz channels:")
    print(f"  {'offset':>8} {'firing frac':>12} {'shift':>9} {'flips':>9}")
    print(f"  {'0.0 K':>8} {float(b_fire[valid].mean()):>12.4f} {'-':>9} {'-':>9}")
    for off in (-3.0, -1.0, 1.0, 3.0):
        o = _outputs(tb, coeffs, dv=off, dh=off)
        w = np.asarray(o.wet, np.float64).reshape(-1)
        v = valid & (w > WET_SENTINELS)
        f = float((w > 0.0)[v].mean())
        fl = float(((w > 0.0) != b_fire)[v].mean())
        res["bias_perturbation"][f"{off:+.1f}K"] = {
            "firing_fraction": f,
            "firing_fraction_shift": f - float(b_fire[v].mean()),
            "classification_flip_frac": fl,
        }
        print(f"  {off:>+7.1f}K {f:>12.4f} {f - float(b_fire[v].mean()):>+9.4f} "
              f"{100*fl:>8.2f}%")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
