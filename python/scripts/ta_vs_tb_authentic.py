"""Which input convention does the recovered decision tree expect?

    python -m scripts.ta_vs_tb_authentic TA_GRID_DIR TB_GRID_DIR SWAMPS_FW.nc
        [--pass dsc] [--out J.json]

The recovered operational source carries antenna-temperature thresholds, while
the modern gridded record supplies brightness temperatures. An earlier test
approximated antenna temperature by subtracting fixed per-channel offsets from
brightness temperature, which made the detector worse but could not settle the
question, because the offsets were an assumption rather than a measurement.

This driver uses authentic antenna temperatures. It composites a month of
gridded antenna temperatures and the matching month of CSU brightness
temperatures on the same grid, runs the same engine over both, and scores both
against the same inundation reference over the cells where both are valid. The
comparison is controlled: identical grid, identical cells, identical reference,
identical thresholds, with only the input convention differing.

It also measures the actual antenna-to-brightness difference per channel, which
the earlier test had to assume.
"""

import json
import os
import sys

import numpy as np

from swi import core_numpy, grids, monthly, validate as val
from swi.channels import N_CHANNELS

FW_INUNDATED = 0.05
FW_CLEAR = 0.10
WET_THR = 0.0

# Grody and Basist (1996) antenna-to-brightness corrections, added to antenna
# temperature to get brightness temperature, in Basist channel order. The
# earlier experiment assumed these; here they are compared with the measured
# difference.
GRODY_OFFSET = np.array([7.0, 7.0, 6.0, 4.0, 4.0, 3.0, 3.0])


def opt(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def _day_key(path):
    import re
    m = re.search(r"D(\d{8})", os.path.basename(path))
    return m.group(1) if m else None


def paired_days(ta_dir, tb_dir):
    """Match antenna and brightness files by date."""
    import glob
    ta = {_day_key(p): p for p in glob.glob(os.path.join(ta_dir, "*.nc"))}
    tb = {_day_key(p): p for p in glob.glob(os.path.join(tb_dir, "*.nc"))}
    days = sorted(set(ta) & set(tb) - {None})
    return [(d, ta[d], tb[d]) for d in days]


def composite(paths, pass_):
    """Monthly mean WET and snow frequency for one input set."""
    # F-13 measures 85.5 GHz directly, so no spectral adjustment applies to
    # either convention and both run on the plain engine.
    comp = monthly.composite(paths, core_numpy, pass_list=(pass_,))
    f = comp["by_pass"][pass_]
    return comp["lat"], comp["lon"], f["wetness_index_mean"], f["snow_frequency"]


def score(wet, snowf, fw_on, land, lat2d):
    """Detection statistics for one composite against the reference."""
    m = land & np.isfinite(fw_on) & (wet >= 0) & ~(snowf > 0.5)
    W, F = wet[m], fw_on[m]
    LAT = lat2d[m]
    w = np.cos(np.deg2rad(LAT))
    s = val.skill_scores(W, F)
    dc = val.weighted_detection_contrast(W, F, w, thr=WET_THR)
    cat = val.categorical(W, F, a_hi=WET_THR, b_hi=FW_INUNDATED)
    hi = F > FW_CLEAR
    return {
        "n_cells": int(s["n"]),
        "spearman_area_weighted": float(val.weighted_spearman(W, F, w)),
        "spearman_unweighted": float(s["spearman_r"]),
        "pearson": float(s["pearson_r"]),
        "contrast_area_weighted": float(dc["ratio"]),
        "firing_fraction": float((W > WET_THR).mean()),
        "clearly_inundated_detected_frac": (
            float((W[hi] > WET_THR).mean()) if hi.sum() > 100 else float("nan")),
        "contingency": {k: (float(v) if isinstance(v, float) else int(v))
                        for k, v in cat.items()},
    }, m


def main():
    argv = sys.argv
    if len(argv) < 4:
        print(__doc__)
        return 1
    ta_dir, tb_dir, swamps = argv[1], argv[2], argv[3]
    pass_ = opt(argv, "--pass", "dsc")
    out_path = opt(argv, "--out", "../results/ta_vs_tb_authentic.json")

    from scripts.validate_swamps import load_swamps_fw

    pairs = paired_days(ta_dir, tb_dir)
    if not pairs:
        print("no matching days between the two directories")
        return 1
    print(f"{len(pairs)} matched days, pass={pass_}")

    ta_paths = [p[1] for p in pairs]
    tb_paths = [p[2] for p in pairs]

    print("compositing antenna temperatures ...")
    lat, lon, wet_ta, snow_ta = composite(ta_paths, pass_)
    print("compositing brightness temperatures ...")
    _, _, wet_tb, snow_tb = composite(tb_paths, pass_)

    slat, slon, fw = load_swamps_fw(swamps)
    fw_on = val.regrid_nearest(slat, slon, fw, lat, lon)
    land = grids.land_mask(lat, lon)
    lat2d = np.broadcast_to(lat[:, None], wet_ta.shape)

    # Score each on the cells where BOTH conventions produced a retrieval, so
    # the two are compared over one identical population.
    # Each convention's own eligible population, reported so the effect of
    # scoring on the intersection is visible rather than hidden.
    base = land & np.isfinite(fw_on)
    elig_ta = int((base & (wet_ta >= 0) & ~(snow_ta > 0.5)).sum())
    elig_tb = int((base & (wet_tb >= 0) & ~(snow_tb > 0.5)).sum())
    both = (base & (wet_ta >= 0) & (wet_tb >= 0)
            & ~(snow_ta > 0.5) & ~(snow_tb > 0.5))
    res_ta, _ = score(np.where(both, wet_ta, np.nan), snow_ta, fw_on, both, lat2d)
    res_tb, _ = score(np.where(both, wet_tb, np.nan), snow_tb, fw_on, both, lat2d)

    # Measured antenna-to-brightness difference per channel over every paired
    # day, and the per-channel spatial correlation between the two gridded
    # inputs, which is the evidence that the gridding is right. Both are saved
    # so neither has to be recomputed to be checked.
    from swi import io_csu_grid as io
    names = ["19v", "19h", "22v", "37v", "37h", "85v", "85h"]
    sums = {n: [] for n in names}
    corrs = {n: [] for n in names}
    print(f"measuring channel offsets over all {len(pairs)} days ...")
    for _, tap, tbp in pairs:
        _, _, A, _ = io.read_channels(tap, pass_=pass_)
        _, _, B, _ = io.read_channels(tbp, pass_=pass_)
        for i, n in enumerate(names):
            a, b = A[:, :, i], B[:, :, i]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 1000:
                continue
            sums[n].append(float(np.mean(b[m] - a[m])))
            corrs[n].append(float(np.corrcoef(a[m], b[m])[0, 1]))
    diffs = {}
    for i, n in enumerate(names):
        mean_off = float(np.mean(sums[n]))
        diffs[n] = {
            "measured_tb_minus_ta": mean_off,
            "grody_assumed": float(GRODY_OFFSET[i]),
            "error_in_assumption": mean_off - float(GRODY_OFFSET[i]),
            "n_days": len(sums[n]),
            "grid_correlation_mean": float(np.mean(corrs[n])),
            "grid_correlation_min": float(np.min(corrs[n])),
        }

    res = {
        "note": ("controlled comparison of input conventions: identical grid, "
                 "identical cells, identical reference and thresholds, with "
                 "only the input convention differing"),
        "pass": pass_,
        "n_days": len(pairs),
        "days": [p[0] for p in pairs],
        "reference": os.path.basename(swamps),
        "n_common_cells": int(both.sum()),
        "antenna_temperature": res_ta,
        "brightness_temperature": res_tb,
        "channel_differences": diffs,
        "offset_and_correlation_basis": (
            "per-day means over every paired day, same pass, over cells where "
            "both gridded inputs are finite"),
        "coverage": {
            "antenna_eligible_cells": int(elig_ta),
            "brightness_eligible_cells": int(elig_tb),
            "scored_intersection_cells": int(both.sum()),
        },
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)

    print(f"\ncommon cells: {int(both.sum()):,}")
    print(f"{'statistic':<34}{'antenna T':>14}{'brightness T':>15}")
    for k, fmt in (("spearman_area_weighted", "{:+.3f}"),
                   ("spearman_unweighted", "{:+.3f}"),
                   ("pearson", "{:+.3f}"),
                   ("contrast_area_weighted", "{:.2f}x"),
                   ("firing_fraction", "{:.3f}"),
                   ("clearly_inundated_detected_frac", "{:.3f}")):
        print(f"  {k:<32}{fmt.format(res_ta[k]):>14}{fmt.format(res_tb[k]):>15}")
    for k in ("POD", "FAR", "CSI", "HSS"):
        print(f"  {k:<32}{res_ta['contingency'][k]:>14.3f}"
              f"{res_tb['contingency'][k]:>15.3f}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
