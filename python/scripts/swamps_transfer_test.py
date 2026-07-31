"""Preregistered SSM/I-era inundation transfer test.

Runs the analysis fixed in `docs/Preregistration_SSMI_Era_SWAMPS_Test.md`: the
revived wetness index for a true-85 GHz SSM/I month against SWAMPS fractional
surface water, with every threshold carried over unchanged from the published
July 2016 analysis rather than retuned. The point is to test whether the single
published month transfers to the other sensor era, so nothing here is fitted.

Reports, per the preregistration and whatever they show: area-weighted and
unweighted Spearman, Pearson, the area-weighted and unweighted detection
contrast, the complete contingency table at the frozen threshold, the rank
correlation among only the cells where the index fires, a 10-degree spatial block
bootstrap interval on the area-weighted Spearman, and the contrast by latitude
zone. Writes every value to JSON.

Usage:
    python -m scripts.swamps_transfer_test PRODUCT.nc SWAMPS_FW.nc [--out J.json]
        [--label TEXT] [--draws N] [--seed N]
"""

import json
import os
import sys

import numpy as np

from swi import grids, validate as val
from scripts.validate_swamps import load_swamps_fw, read_product

# Frozen in the preregistration. Not tunable from the command line on purpose.
WET_THR = 0.0            # the index "fires"
FW_INUNDATED = 0.05      # reference calls a cell inundated
FW_CLEAR = 0.10          # reference calls a cell clearly inundated
BLOCK_DEG = 10.0


def opt(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def main():
    argv = sys.argv
    if len(argv) < 3:
        print(__doc__)
        return 1
    product, swamps = argv[1], argv[2]
    out_path = opt(argv, "--out", "../scratch/swamps_transfer_test.json")
    label = opt(argv, "--label", os.path.basename(product))
    draws = int(opt(argv, "--draws", "2000"))
    seed = int(opt(argv, "--seed", "20260727"))

    plat, plon, wet, snowf = read_product(product)
    slat, slon, fw = load_swamps_fw(swamps)
    fw_on = val.regrid_nearest(slat, slon, fw, plat, plon)

    land = grids.land_mask(plat, plon)
    if land.all():
        raise RuntimeError(
            "land mask is all-True, so no ocean was excluded. Install "
            "global_land_mask; these statistics are land-only by definition.")
    m = land & np.isfinite(fw_on) & (wet >= 0) & ~(snowf > 0.5)
    lat2d = np.broadcast_to(plat[:, None], wet.shape)
    lon2d = np.broadcast_to(plon[None, :], wet.shape)
    W, F = wet[m], fw_on[m]
    LAT, LON = lat2d[m], lon2d[m]
    w = np.cos(np.deg2rad(LAT))          # area weight on an equal-angle grid

    s = val.skill_scores(W, F)
    sp_w = val.weighted_spearman(W, F, w)
    sp_w_rd = val.weighted_spearman_rankdist(W, F, w)
    pe_w = val.weighted_pearson(W, F, w)
    dc = val.detection_contrast(W, F, thr=WET_THR)
    dc_w = val.weighted_detection_contrast(W, F, w, thr=WET_THR)
    cat = val.categorical(W, F, a_hi=WET_THR, b_hi=FW_INUNDATED)

    hi = F > FW_CLEAR
    clear_hit = float((W[hi] > WET_THR).mean()) if hi.sum() > 100 else float("nan")

    pos = W > WET_THR
    pos_stats = val.skill_scores(W[pos], F[pos]) if pos.sum() > 100 else {"spearman_r": float("nan")}

    def wsp(idx):
        # Re-rank inside each draw. Ranking once outside and resampling the
        # fixed ranks holds the estimator's own variability constant and gives
        # an interval that is too narrow for the statistic it names.
        return val.weighted_spearman(W[idx], F[idx], w[idx])

    ci = val.block_bootstrap_ci(wsp, val.block_ids(LAT, LON, BLOCK_DEG),
                                n_draws=draws, seed=seed, return_draws=True)

    def wcontrast(idx):
        r = val.weighted_detection_contrast(W[idx], F[idx], w[idx], thr=WET_THR)
        return r["ratio"]
    ci_c = val.block_bootstrap_ci(wcontrast, val.block_ids(LAT, LON, BLOCK_DEG),
                                  n_draws=draws, seed=seed)

    zones = val.detection_by_zone(W, F, LAT, thr=WET_THR)

    res = {
        "label": label, "product": product, "reference": swamps,
        "frozen_thresholds": {"wet": WET_THR, "fw_inundated": FW_INUNDATED,
                              "fw_clearly_inundated": FW_CLEAR},
        "n_cells": int(s["n"]),
        "spearman_area_weighted": sp_w, "spearman_unweighted": s["spearman_r"],
        "spearman_area_weighted_rankdist": sp_w_rd,
        "pearson_area_weighted": pe_w, "pearson_unweighted": s["pearson_r"],
        "contrast_area_weighted": dc_w, "contrast_unweighted": dc,
        "contingency": cat,
        "clearly_inundated_detected_frac": clear_hit,
        "n_clearly_inundated": int(hi.sum()),
        "spearman_among_firing_cells": pos_stats["spearman_r"],
        "n_firing_cells": int(pos.sum()),
        "block_bootstrap_area_weighted_spearman": {
            "block_deg": BLOCK_DEG, "seed": seed, **ci},
        "block_bootstrap_area_weighted_contrast": {
            "block_deg": BLOCK_DEG, "seed": seed, **ci_c},
        "contrast_by_zone": zones,
    }

    print(f"\n{label}: revived index vs SWAMPS fractional water, land only "
          f"(n={s['n']:,})")
    print(f"  Spearman  area-weighted {sp_w:+.3f}   unweighted {s['spearman_r']:+.3f}")
    print(f"  Pearson   area-weighted {pe_w:+.3f}   unweighted {s['pearson_r']:+.3f}")
    print(f"  block bootstrap 95% CI on area-weighted Spearman: "
          f"[{ci['lo']:+.3f}, {ci['hi']:+.3f}] over {ci['n_blocks']} blocks, "
          f"P(>0)={ci['frac_gt_0']:.3f}")
    print(f"  contrast  area-weighted {dc_w['ratio']:.2f}x   "
          f"unweighted {dc['ratio']:.2f}x  (n_fire={dc['n_hi']:,})")
    print(f"  contingency at frozen threshold: POD={cat['POD']:.2f} "
          f"FAR={cat['FAR']:.2f} CSI={cat['CSI']:.2f} HSS={cat['HSS']:.2f}")
    print(f"  clearly inundated (fw>{FW_CLEAR}) detected: {100*clear_hit:.0f}% "
          f"of {int(hi.sum()):,}")
    print(f"  Spearman among firing cells only: "
          f"{pos_stats['spearman_r']:+.3f} (n={int(pos.sum()):,})")
    print("  contrast by zone:")
    for name, z in zones.items():
        print(f"    {name:<14}: {z['ratio']:.2f}x  (n_fire={z['n_hi']:,})")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
