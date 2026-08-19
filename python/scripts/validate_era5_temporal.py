"""Per-cell temporal anomaly validation: does WET track soil-moisture changes?

    python -m scripts.validate_era5_temporal OUT.png [--min-n N] PRODUCT_YYYYMM.nc ...

For each monthly product, loads ERA5-Land swvl1, regrids to our grid, masks to
unfrozen land, and stacks the months. Then computes the per-cell temporal
correlation of the two monthly series. Note that this is NOT an anomaly correlation:
subtracting each cell's own temporal mean leaves a correlation unchanged, so the
seasonal cycle is retained in both series and this measures mainly whether the index
tracks the annual wet and dry march. Removing a calendar-month climatology would be
needed to test skill at interannual departures.

--min-n sets the minimum valid months a cell needs to be evaluated (default
max(6, n - 3), tuned for a single year; for a multi-year run set it to roughly
two thirds of the months so seasonally-frozen cells are not all excluded).
"""

import os
import sys

import numpy as np

from swi import figcache
from swi import validate as val
from swi.io_era5 import load_swvl1


def read_product(path, pass_="dsc"):
    import netCDF4 as nc
    ds = nc.Dataset(path)
    try:
        lat = np.asarray(ds["lat"][:], np.float64)
        lon = np.asarray(ds["lon"][:], np.float64)
        wet = np.ma.filled(ds[f"wetness_index_mean_{pass_}"][:], np.nan).astype(np.float64)
        snowf = np.ma.filled(ds[f"snow_frequency_{pass_}"][:], np.nan).astype(np.float64)
        month = str(getattr(ds, "month", ""))
    finally:
        ds.close()
    return lat, lon, wet, snowf, month


def main():
    if "--from-cache" in sys.argv:
        cpath = sys.argv[sys.argv.index("--from-cache") + 1]
        out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv \
            else sys.argv[sys.argv.index("--from-cache") + 2]
        arrays, _ = figcache.load(cpath)
        _map(arrays["lat"], arrays["lon"], arrays["r"], out)
        return 0

    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    args = sys.argv[1:]
    min_n = None
    json_out = None
    if "--min-n" in args:
        i = args.index("--min-n")
        min_n = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    if "--json" in args:
        i = args.index("--json"); json_out = args[i + 1]; args = args[:i] + args[i + 2:]
    no_cache = "--no-cache" in args
    args = [a for a in args if a != "--no-cache"]
    out_png = args[0]
    products = sorted(args[1:])

    os.makedirs("../data/era5", exist_ok=True)
    wet_stack, sm_stack = [], []
    plat = plon = None
    months = []
    for p in products:
        lat, lon, wet, snowf, month = read_product(p)
        plat, plon = lat, lon
        elat, elon, sm = load_swvl1(month, f"../data/era5/era5land_swvl1_{month}.nc")
        sm_on = val.regrid_nearest(elat, elon, sm, plat, plon)
        valid = np.isfinite(sm_on) & (wet >= 0) & ~(snowf > 0.5)
        wet_stack.append(np.where(valid, wet, np.nan))
        sm_stack.append(np.where(valid, sm_on, np.nan))
        months.append(month)
        print(f"  {month}: {int(valid.sum()):,} valid land cells")

    A = np.stack(wet_stack); B = np.stack(sm_stack)
    if min_n is None:
        min_n = max(6, len(products) - 3)
    r, n = val.temporal_anomaly_correlation(A, B, min_n=min_n)
    print(f"\n(min valid months per cell = {min_n} of {len(products)})")
    g = r[np.isfinite(r)]
    print(f"\nTemporal correlation (WET vs ERA5-Land swvl1), "
          f"{len(products)} months {months[0]}..{months[-1]}:")
    print(f"  cells evaluated   : {g.size:,}")
    print(f"  median r          : {np.median(g):+.3f}")
    print(f"  mean r            : {np.mean(g):+.3f}")
    print(f"  fraction r > 0    : {100*np.mean(g > 0):.1f}%")
    print(f"  fraction r > 0.3  : {100*np.mean(g > 0.3):.1f}%")
    print(f"  fraction r > 0.5  : {100*np.mean(g > 0.5):.1f}%")

    if json_out:
        import json
        res = {
            "reference": "ERA5-Land swvl1, per-cell temporal correlation",
            "note": "seasonal cycle retained in both series; not an anomaly "
                    "correlation (subtracting a per-cell constant leaves a "
                    "correlation unchanged)",
            "months": months,
            "n_months": len(products),
            "min_valid_months_per_cell": int(min_n),
            "cells_evaluated": int(g.size),
            "median_r": float(np.median(g)),
            "mean_r": float(np.mean(g)),
            "fraction_positive": float(np.mean(g > 0)),
            "fraction_above_0.3": float(np.mean(g > 0.3)),
            "fraction_above_0.5": float(np.mean(g > 0.5)),
        }
        os.makedirs(os.path.dirname(json_out) or ".", exist_ok=True)
        with open(json_out, "w") as f:
            json.dump(res, f, indent=2, sort_keys=True)
        print(f"wrote {json_out}")

    # Plot the float32-stored arrays so a --from-cache redraw is identical,
    # and save the cache only after a successful render.
    stored = figcache.as_stored(lat=plat, lon=plon, r=r)
    _map(stored["lat"], stored["lon"], stored["r"], out_png)
    if not no_cache:
        figcache.save(figcache.cache_path("fig_temporal"), {
            "figure": "fig_temporal",
            "script": "python/scripts/validate_era5_temporal.py",
            "months": months,
            "n_months": len(products),
            "min_valid_months_per_cell": int(min_n),
        }, **stored)
    return 0


def _map(lat, lon, r, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Author at the ~6.5 in text width so titles/labels land >= 8 pt on the page
    # after the figure-to-page reduction (paper-figure-standards.md, section 2).
    w = 6.5; sc = w / 6.5
    tsz, lab, tick = 10 * sc, 10 * sc, 9 * sc
    roll = lon.size // 2
    img = np.roll(r, roll, axis=1)
    fig, ax = plt.subplots(figsize=(w, 3.0))
    im = ax.imshow(img, origin="lower", extent=[-180, 180, lat.min(), lat.max()],
                   aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("WET vs ERA5-Land soil moisture, per-cell temporal correlation\n"
                 "(seasonal cycle retained)", fontsize=tsz, pad=4)
    ax.set_xlabel("longitude", fontsize=lab); ax.set_ylabel("latitude", fontsize=lab)
    ax.tick_params(labelsize=tick)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, label="per-cell temporal correlation r")
    cb.set_label("per-cell temporal correlation r", fontsize=lab)
    cb.ax.tick_params(labelsize=tick)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
