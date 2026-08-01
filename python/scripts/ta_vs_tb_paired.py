"""Which input convention does the recovered decision tree expect, controlled.

    python -m scripts.ta_vs_tb_paired PAIRED_GRID_DIR SWAMPS_FW.nc
        [--pass dsc] [--out J.json]

This supersedes `ta_vs_tb_authentic`, which composited a self-gridded antenna
field and the official CSU brightness grid separately and scored them against
each other. Those two were built by different spatial operators, so the measured
separation carried the gridding as well as the input convention. Independent
review blocked that construction, and the numbers it produced should not be
quoted.

Here both arms come from the paired daily grids written by
`scripts.grid_paired_day`, in which antenna and brightness temperature were
binned through one geolocation, one set of scan times, one quality screen, one
pass assignment and one set of contributing pixels per cell. See
`swi/paired_swath.py` for why that is possible and what it costs. Every cell
scored therefore rests on identical observations in the two arms, and the only
thing differing between them is the radiometric convention.

Both arms are also required to be valid on the same days before a month is
composited, which the superseded driver did not do. Compositing independently
and intersecting only the final monthly masks let one cell rest on a different
number of contributing days in the two arms, and the monthly operating point
depends on how many days contributed.
"""

import json
import os
import sys

import numpy as np

from scripts.ta_vs_tb_authentic import (FW_CLEAR, FW_INUNDATED, GRODY_OFFSET,
                                        WET_THR, opt, score)
from swi import core_numpy, grids, io_csu_grid as io, monthly
from swi import validate as val
from swi.channels import N_CHANNELS

TA_SENSOR = "ssmi_ta"       # the antenna arm of a paired file
TB_SENSOR = "ssmi"          # the brightness arm of the same file


def day_files(paired_dir):
    import glob
    return sorted(glob.glob(os.path.join(paired_dir, "SWI_PAIRED_*.nc")))


def common_support_days(paths, pass_):
    """Per-day masks where both arms return a retrieval, and the day counts.

    A day contributes to a cell only when both conventions produced a valid
    retrieval there, so the two monthly composites rest on the same days as well
    as the same cells.
    """
    masks, counts = [], None
    for p in paths:
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                a = io.evaluate_file(p, pass_=pass_, engine=core_numpy,
                                     sensor=TA_SENSOR)
                b = io.evaluate_file(p, pass_=pass_, engine=core_numpy,
                                     sensor=TB_SENSOR)
        except Exception:
            masks.append(None)
            continue
        m = (a["valid"] & b["valid"]
             & (a["wet"] >= 0) & (b["wet"] >= 0))
        counts = m.astype(np.int32) if counts is None else counts + m
        masks.append(m)
    return masks, counts


def composite_arm(paths, pass_, sensor, day_masks):
    """Composite one arm over the days both arms retrieve.

    Uses the project's own monthly accumulator rather than a plain mean, because
    snow frequency counts observed days against flagged days and has to respect
    the fill and gap sentinels. Cells outside a day's shared mask are set to the
    gap sentinel so that day does not count as observed there.
    """
    acc = None
    lat = lon = None
    used = 0
    for path, keep in zip(paths, day_masks):
        if keep is None:
            continue
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                r = io.evaluate_file(path, pass_=pass_, engine=core_numpy,
                                     sensor=sensor)
        except Exception:
            continue
        if r.get("empty_channels"):
            continue
        if lat is None:
            lat, lon = r["lat"], r["lon"]
        if acc is None:
            acc = monthly.Accumulator(r["wet"].shape)
        snow = np.asarray(r["snow"])
        acc.add({
            "wet": np.where(keep, r["wet"], np.nan),
            "temp": np.where(keep, r["temp"], -99.0),
            "snow": np.where(keep, snow, -100).astype(snow.dtype),
        })
        used += 1
    if acc is None:
        raise SystemExit("no day of this arm could be read")
    res = acc.result()
    return (lat, lon, res["wetness_index_mean"], res["snow_frequency"], used,
            res)


def channel_differences(paths, pass_):
    """Measured brightness minus antenna per channel, on the shared cells."""
    out = {}
    sums = np.zeros(N_CHANNELS)
    n = 0
    for p in paths:
        try:
            _, _, ta, _ = io.read_channels(p, pass_=pass_, sensor=TA_SENSOR)
            _, _, tb, _ = io.read_channels(p, pass_=pass_, sensor=TB_SENSOR)
        except Exception:
            continue
        ok = np.isfinite(ta).all(axis=2) & np.isfinite(tb).all(axis=2)
        if not ok.any():
            continue
        sums += (tb - ta)[ok].mean(axis=0)
        n += 1
    names = ["19v", "19h", "22v", "37v", "37h", "85v", "85h"]
    for i, nm in enumerate(names):
        measured = float(sums[i] / n) if n else float("nan")
        out[nm] = {
            "measured_tb_minus_ta": measured,
            "grody_assumed": float(GRODY_OFFSET[i]),
            "error_in_assumption": measured - float(GRODY_OFFSET[i]),
            "n_days": n,
        }
    return out


def main():
    argv = sys.argv
    if len(argv) < 3:
        print(__doc__)
        return 1
    paired_dir, swamps = argv[1], argv[2]
    pass_ = opt(argv, "--pass", "dsc")
    out_path = opt(argv, "--out", "../results/ta_vs_tb_paired.json")

    from scripts.validate_swamps import load_swamps_fw

    paths = day_files(paired_dir)
    if not paths:
        print(f"no paired grids in {paired_dir}")
        return 1
    print(f"{len(paths)} paired days, pass={pass_}")

    print("finding days where both arms retrieve ...")
    masks, day_counts = common_support_days(paths, pass_)

    print("compositing the antenna arm ...")
    lat, lon, wet_ta, snow_ta, n_ta, full_ta = composite_arm(
        paths, pass_, TA_SENSOR, masks)
    print("compositing the brightness arm ...")
    _, _, wet_tb, snow_tb, n_tb, full_tb = composite_arm(
        paths, pass_, TB_SENSOR, masks)

    slat, slon, fw = load_swamps_fw(swamps)
    fw_on = val.regrid_nearest(slat, slon, fw, lat, lon)
    land = grids.land_mask(lat, lon)
    lat2d = np.broadcast_to(lat[:, None], wet_ta.shape)

    base = land & np.isfinite(fw_on)
    elig_ta = int((base & (wet_ta >= 0) & ~(snow_ta > 0.5)).sum())
    elig_tb = int((base & (wet_tb >= 0) & ~(snow_tb > 0.5)).sum())
    both = (base & (wet_ta >= 0) & (wet_tb >= 0)
            & ~(snow_ta > 0.5) & ~(snow_tb > 0.5))
    res_ta, _ = score(np.where(both, wet_ta, np.nan), snow_ta, fw_on, both,
                      lat2d)
    res_tb, _ = score(np.where(both, wet_tb, np.nan), snow_tb, fw_on, both,
                      lat2d)

    union = elig_ta + elig_tb - int(both.sum())
    contributing = day_counts[both] if day_counts is not None else np.array([0])
    result = {
        "antenna_temperature": res_ta,
        "brightness_temperature": res_tb,
        "channel_differences": channel_differences(paths, pass_),
        "coverage": {
            "antenna_eligible_cells": elig_ta,
            "brightness_eligible_cells": elig_tb,
            "scored_intersection_cells": int(both.sum()),
            "union_cells": union,
            "intersection_over_antenna": int(both.sum()) / max(elig_ta, 1),
            "intersection_over_brightness": int(both.sum()) / max(elig_tb, 1),
            "intersection_over_union": int(both.sum()) / max(union, 1),
        },
        "contributing_days": {
            "mean": float(contributing.mean()),
            "min": int(contributing.min()),
            "max": int(contributing.max()),
            "note": ("identical for both arms by construction, since a day "
                     "enters a cell only when both conventions retrieve there"),
        },
        "wet_frequency_on_scored_cells": {
            "antenna_mean": float(np.nanmean(full_ta["wet_frequency"][both])),
            "brightness_mean": float(np.nanmean(full_tb["wet_frequency"][both])),
            "note": ("fraction of a cell's contributing days on which the "
                     "index fired, carried so a frequency-based monthly "
                     "operating point can be tested without regridding"),
        },
        "n_common_cells": int(both.sum()),
        "n_days": len(paths),
        "days_composited": {"antenna": n_ta, "brightness": n_tb},
        "pass": pass_,
        "reference": os.path.basename(swamps),
        "note": ("controlled comparison of input conventions. Both arms are "
                 "binned from matched swath granules through one geolocation, "
                 "one scan-time selection, one quality screen and one set of "
                 "contributing pixels per cell, and a day enters a cell only "
                 "when both arms retrieve there. Only the radiometric "
                 "convention differs. Neither arm reproduces the official CSU "
                 "daily grid, which uses a footprint-aware resampler."),
        "supersedes": "ta_vs_tb_authentic_F13_199807.json",
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(f"\nwrote {out_path}")
    for label, r in (("antenna", res_ta), ("brightness", res_tb)):
        print(f"  {label:11s} spearman {r['spearman_area_weighted']:.4f}  "
              f"contrast {r['contrast_area_weighted']:.4f}  "
              f"firing {r['firing_fraction']:.4f}  "
              f"CSI {r['contingency']['CSI']:.4f}  "
              f"HSS {r['contingency']['HSS']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
