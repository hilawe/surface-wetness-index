"""Categorical skill across index thresholds, with cross-month transfer.

    python -m scripts.threshold_transfer_test PRODUCT_A.nc SWAMPS_A.nc LABEL_A \
        PRODUCT_B.nc SWAMPS_B.nc LABEL_B [--out J.json] [--png F.png]

The paper's categorical scores use the natural operating point, any positive
monthly index, which was never tuned. This driver sweeps the index threshold in
each month against the same inundation definition (fractional water above 0.05),
reports the precision-recall curve, finds the threshold maximizing the critical
success index in each month, and scores each month's best threshold in the
other month. The sweep answers whether the natural threshold's false alarm
ratio is intrinsic to the index or a property of the untuned operating point,
and the transfer shows whether a tuned threshold carries across sensor eras.

This analysis is post hoc and was not part of the preregistered comparisons.
"""

import json
import os
import sys

import numpy as np

from swi import grids, validate as val


def opt(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


FW_INUNDATED = 0.05
# Uniform 0.5-step grid. A coarser grid placed the optimum at a grid point
# rather than at the true maximum, so the step is fixed and reported.
THRESHOLD_STEP = 0.5
THRESHOLDS = [round(0.5 * i, 1) for i in range(0, 61)]


def read_product(path, pass_="dsc"):
    import netCDF4 as nc
    ds = nc.Dataset(path)
    try:
        lat = np.asarray(ds["lat"][:], np.float64)
        lon = np.asarray(ds["lon"][:], np.float64)
        wet = np.ma.filled(ds[f"wetness_index_mean_{pass_}"][:], np.nan).astype(np.float64)
        snowf = np.ma.filled(ds[f"snow_frequency_{pass_}"][:], np.nan).astype(np.float64)
    finally:
        ds.close()
    return lat, lon, wet, snowf


def month_population(product, swamps):
    from scripts.validate_swamps import load_swamps_fw
    plat, plon, wet, snowf = read_product(product)
    slat, slon, fw = load_swamps_fw(swamps)
    fw_on = val.regrid_nearest(slat, slon, fw, plat, plon)
    land = grids.land_mask(plat, plon)
    m = land & np.isfinite(fw_on) & (wet >= 0) & ~(snowf > 0.5)
    return wet[m], fw_on[m]


def sweep(W, F):
    rows = []
    for t in THRESHOLDS:
        c = val.categorical(W, F, a_hi=t, b_hi=FW_INUNDATED)
        rows.append({"threshold": t, "POD": c["POD"], "FAR": c["FAR"],
                     "precision": 1.0 - c["FAR"], "CSI": c["CSI"],
                     "HSS": c["HSS"]})
    best = max(rows, key=lambda r: (r["CSI"] if np.isfinite(r["CSI"]) else -1))
    return rows, best


def main():
    argv = sys.argv
    if len(argv) < 7:
        print(__doc__)
        return 1
    pa, sa, la, pb, sb, lb = argv[1:7]
    out_path = opt(argv, "--out", "../results/threshold_transfer.json")
    png = opt(argv, "--png")

    Wa, Fa = month_population(pa, sa)
    Wb, Fb = month_population(pb, sb)
    rows_a, best_a = sweep(Wa, Fa)
    rows_b, best_b = sweep(Wb, Fb)

    # Cross-month transfer of each month's best-CSI threshold.
    xfer_ab = val.categorical(Wb, Fb, a_hi=best_a["threshold"], b_hi=FW_INUNDATED)
    xfer_ba = val.categorical(Wa, Fa, a_hi=best_b["threshold"], b_hi=FW_INUNDATED)

    res = {
        "note": ("post hoc threshold sweep, not part of the preregistered "
                 "comparisons. The reported optimum is the best threshold on "
                 "the stated grid, not a continuous maximum. When the two "
                 "optima coincide, each cross-score equals the receiving "
                 "month's own tuned score by construction."),
        "threshold_step": THRESHOLD_STEP,
        "inundation_definition": f"fractional water > {FW_INUNDATED}",
        "thresholds": THRESHOLDS,
        la: {"n": int(Wa.size), "curve": rows_a, "best_csi": best_a},
        lb: {"n": int(Wb.size), "curve": rows_b, "best_csi": best_b},
        "transfer": {
            f"{la}_best_scored_in_{lb}": {
                "threshold": best_a["threshold"],
                "POD": xfer_ab["POD"], "FAR": xfer_ab["FAR"],
                "CSI": xfer_ab["CSI"], "HSS": xfer_ab["HSS"]},
            f"{lb}_best_scored_in_{la}": {
                "threshold": best_b["threshold"],
                "POD": xfer_ba["POD"], "FAR": xfer_ba["FAR"],
                "CSI": xfer_ba["CSI"], "HSS": xfer_ba["HSS"]},
        },
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)

    for lab, rows, best in ((la, rows_a, best_a), (lb, rows_b, best_b)):
        print(f"\n{lab}:")
        print(f"  {'thr':>5} {'POD':>6} {'FAR':>6} {'CSI':>6} {'HSS':>6}")
        for r in rows:
            mark = "  <- best CSI" if r is best else ""
            print(f"  {r['threshold']:>5.1f} {r['POD']:>6.2f} {r['FAR']:>6.2f} "
                  f"{r['CSI']:>6.2f} {r['HSS']:>6.2f}{mark}")
    print(f"\ntransfer {la} best (t={best_a['threshold']}) in {lb}: "
          f"POD {xfer_ab['POD']:.2f} FAR {xfer_ab['FAR']:.2f} CSI {xfer_ab['CSI']:.2f}")
    print(f"transfer {lb} best (t={best_b['threshold']}) in {la}: "
          f"POD {xfer_ba['POD']:.2f} FAR {xfer_ba['FAR']:.2f} CSI {xfer_ba['CSI']:.2f}")
    print(f"wrote {out_path}")

    if png:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["axes.unicode_minus"] = False
        import matplotlib.pyplot as plt
        w = 7.5
        sc = w / 6.5
        t_, lab_, tick = 10 * sc, 10 * sc, 9 * sc
        fig, ax = plt.subplots(figsize=(w, 4.2))
        def disp(lab):
            return (lab.replace("_", ", ").replace("F16", "F-16")
                    .replace("F13", "F-13").replace("July", "July "))
        for lab, rows, best, color in ((disp(la), rows_a, best_a, "C0"),
                                       (disp(lb), rows_b, best_b, "C1")):
            pod = [r["POD"] for r in rows]
            prec = [r["precision"] for r in rows]
            ax.plot(pod, prec, "o-", ms=4, lw=1.4, color=color, label=lab)
            ax.plot(rows[0]["POD"], rows[0]["precision"], "s", ms=9,
                    mfc="none", mec=color, mew=1.6)
            ax.plot(best["POD"], best["precision"], "*", ms=14, color=color)
        ax.plot([], [], "s", ms=9, mfc="none", mec="0.3", mew=1.6,
                label="natural threshold, WET > 0")
        ax.plot([], [], "*", ms=12, color="0.3", label="best CSI threshold")
        ax.set_xlabel("probability of detection", fontsize=lab_)
        ax.set_ylabel("precision (1 - false alarm ratio)", fontsize=lab_)
        ax.set_title("Categorical skill across index thresholds, "
                     "fractional water above 0.05", fontsize=t_, pad=5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=tick, loc="lower left", frameon=False)
        ax.tick_params(labelsize=tick)
        fig.tight_layout()
        fig.savefig(png, dpi=150)
        print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
