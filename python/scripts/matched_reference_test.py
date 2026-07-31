"""Detection contrast against two soil-moisture references on identical cells.

    python -m scripts.matched_reference_test USCRN_DIR PRODUCT_YYYYMM.nc ...
        [--out J.json]

The paper reports a wet versus dry detection contrast near 1.4 against in-situ
USCRN stations, the ERA5-Land reanalysis, and the ESA CCI microwave record. Those
three numbers come from different domains, periods, and validity masks, so their
agreement is suggestive rather than a controlled comparison.

This driver removes the support mismatch between two of them. It takes the
station-months where both references and the retrieval carry valid values, reads
ERA5-Land at exactly the grid cells containing those stations for exactly those
months, and computes the detection contrast for both references over that
identical population. A
station-block bootstrap gives an interval for each contrast and, more to the
point, for their difference, since the two are measured on the same draws.

A residual mismatch remains and is not removable, since USCRN is a point
measurement and ERA5-Land is a cell average, so the two describe the same cells
at different spatial scales.
"""

import json
import os
import sys

import numpy as np

from swi import validate as val
from swi.io_era5 import load_swvl1
from swi.io_uscrn import load_station_monthly


def opt(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def read_product(path, pass_="dsc"):
    import netCDF4 as nc
    ds = nc.Dataset(path)
    try:
        lat = np.asarray(ds["lat"][:], np.float64)
        lon = np.asarray(ds["lon"][:], np.float64)
        wet = np.ma.filled(ds[f"wetness_index_mean_{pass_}"][:], np.nan).astype(np.float64)
        snowf = np.ma.filled(ds[f"snow_frequency_{pass_}"][:], np.nan).astype(np.float64)
        month = str(getattr(ds, "month", os.path.basename(path)))
    finally:
        ds.close()
    return lat, lon, wet, snowf, month


def main():
    argv = sys.argv
    if len(argv) < 3:
        print(__doc__)
        return 1
    args = argv[1:]
    out_path = opt(args, "--out", "../results/matched_reference_test.json")
    if "--out" in args:
        i = args.index("--out"); args = args[:i] + args[i + 2:]
    uscrn_dir = args[0]
    products = sorted(args[1:])

    stations = load_station_monthly(uscrn_dir)
    if not stations:
        print(f"no USCRN station data found in {uscrn_dir}")
        return 1

    W, U, E, SID = [], [], [], []
    months_used = set()
    for prod in products:
        lat, lon, wet, snowf, month = read_product(prod)
        era5_path = f"../data/era5/era5land_swvl1_{month}.nc"
        if not os.path.exists(era5_path):
            print(f"  {month}: no ERA5 file, skipped")
            continue
        elat, elon, sm = load_swvl1(month, era5_path)
        sm_on = val.regrid_nearest(elat, elon, sm, lat, lon)
        n_month = 0
        for wban, st in stations.items():
            u = st["months"].get(month)
            if u is None:
                continue
            i = int(np.argmin(np.abs(lat - st["lat"])))
            j = int(np.argmin(np.abs(lon - (st["lon"] % 360.0))))
            wv, sf, ev = wet[i, j], snowf[i, j], sm_on[i, j]
            # Identical mask for both references: the cell must give a valid
            # retrieval and both references must have a value there.
            if not np.isfinite(wv) or wv < 0 or sf > 0.5 or not np.isfinite(ev):
                continue
            W.append(wv); U.append(u); E.append(ev); SID.append(wban)
            months_used.add(month)
            n_month += 1
        print(f"  {month}: {n_month} matched station-months")

    W = np.asarray(W); U = np.asarray(U); E = np.asarray(E)
    _, sid_codes = np.unique(np.asarray(SID), return_inverse=True)
    if W.size < 50:
        print("too few matched station-months")
        return 1

    dc_u = val.detection_contrast(W, U, thr=0.0)
    dc_e = val.detection_contrast(W, E, thr=0.0)

    def c_uscrn(idx):
        return val.detection_contrast(W[idx], U[idx], thr=0.0)["ratio"]

    def c_era5(idx):
        return val.detection_contrast(W[idx], E[idx], thr=0.0)["ratio"]

    def c_diff(idx):
        return c_uscrn(idx) - c_era5(idx)

    ci_u = val.block_bootstrap_ci(c_uscrn, sid_codes, n_draws=2000,
                                  seed=20260731, null_value=1.0)
    ci_e = val.block_bootstrap_ci(c_era5, sid_codes, n_draws=2000,
                                  seed=20260731, null_value=1.0)
    ci_d = val.block_bootstrap_ci(c_diff, sid_codes, n_draws=2000, seed=20260731)

    res = {
        "note": ("post hoc matched-support comparison of two soil-moisture "
                 "references over identical station-months. USCRN is a point "
                 "measurement and ERA5-Land a cell average, a scale difference "
                 "this design does not remove."),
        "n_station_months": int(W.size),
        "n_stations": int(sid_codes.max() + 1),
        "n_months": len(months_used),
        "uscrn": {"contrast": float(dc_u["ratio"]),
                  "mean_hi": float(dc_u["mean_hi"]),
                  "mean_lo": float(dc_u["mean_lo"]),
                  "n_hi": int(dc_u["n_hi"]),
                  "ci_station_blocks": {k: (float(v) if isinstance(v, float) else int(v))
                                        for k, v in ci_u.items() if k != "draws"}},
        "era5_land": {"contrast": float(dc_e["ratio"]),
                      "mean_hi": float(dc_e["mean_hi"]),
                      "mean_lo": float(dc_e["mean_lo"]),
                      "n_hi": int(dc_e["n_hi"]),
                      "ci_station_blocks": {k: (float(v) if isinstance(v, float) else int(v))
                                            for k, v in ci_e.items() if k != "draws"}},
        "difference_uscrn_minus_era5": {
            "point": float(dc_u["ratio"] - dc_e["ratio"]),
            "ci_station_blocks": {k: (float(v) if isinstance(v, float) else int(v))
                                  for k, v in ci_d.items() if k != "draws"},
            "spans_zero": bool(ci_d["lo"] < 0.0 < ci_d["hi"]),
        },
        "seed": 20260731,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)

    print(f"\nMatched population: {W.size:,} station-months, "
          f"{sid_codes.max() + 1} stations, {len(months_used)} months")
    print(f"  contrast vs USCRN in-situ : {dc_u['ratio']:.2f}  "
          f"[{ci_u['lo']:.2f}, {ci_u['hi']:.2f}]")
    print(f"  contrast vs ERA5-Land     : {dc_e['ratio']:.2f}  "
          f"[{ci_e['lo']:.2f}, {ci_e['hi']:.2f}]")
    print(f"  difference (USCRN - ERA5) : "
          f"{dc_u['ratio'] - dc_e['ratio']:+.2f}  "
          f"[{ci_d['lo']:+.2f}, {ci_d['hi']:+.2f}]  "
          f"spans zero: {res['difference_uscrn_minus_era5']['spans_zero']}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
