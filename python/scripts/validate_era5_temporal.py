"""Per-cell temporal anomaly validation: does WET track soil-moisture changes?

    python -m scripts.validate_era5_temporal OUT.png [--min-n N] PRODUCT_YYYYMM.nc ...

For each monthly product, loads ERA5-Land swvl1, regrids to our grid, masks to
unfrozen land, and stacks the months. Then computes the per-cell temporal anomaly
correlation (departures from each cell's own temporal mean). This asks whether the
wetness index follows the wetness cycle at each location, which is a stronger test
than a single-month spatial comparison.

--min-n sets the minimum valid months a cell needs to be evaluated (default
max(6, n - 3), tuned for a single year; for a multi-year run set it to roughly
two thirds of the months so seasonally-frozen cells are not all excluded).
"""

import os
import sys

import numpy as np

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
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    args = sys.argv[1:]
    min_n = None
    if "--min-n" in args:
        i = args.index("--min-n")
        min_n = int(args[i + 1])
        args = args[:i] + args[i + 2:]
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
    print(f"\nTemporal anomaly correlation (WET vs ERA5-Land swvl1), "
          f"{len(products)} months {months[0]}..{months[-1]}:")
    print(f"  cells evaluated   : {g.size:,}")
    print(f"  median r          : {np.median(g):+.3f}")
    print(f"  mean r            : {np.mean(g):+.3f}")
    print(f"  fraction r > 0    : {100*np.mean(g > 0):.1f}%")
    print(f"  fraction r > 0.3  : {100*np.mean(g > 0.3):.1f}%")
    print(f"  fraction r > 0.5  : {100*np.mean(g > 0.5):.1f}%")

    _map(plat, plon, r, out_png, months)
    return 0


def _map(lat, lon, r, out, months):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    roll = lon.size // 2
    img = np.roll(r, roll, axis=1)
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(img, origin="lower", extent=[-180, 180, lat.min(), lat.max()],
                   aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(f"WET vs ERA5-Land soil moisture: temporal anomaly correlation "
                 f"({months[0]} to {months[-1]})")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    fig.colorbar(im, ax=ax, shrink=0.85, label="per-cell anomaly correlation r")
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
