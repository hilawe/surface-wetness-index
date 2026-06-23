"""Validate monthly WET products against USCRN 5 cm soil moisture (point to pixel).

    python -m scripts.validate_uscrn USCRN_DIR PRODUCT_YYYYMM.nc ...

For each monthly product, every USCRN station with a monthly 5 cm soil-moisture
mean for that month is co-located with the wetness index at the grid cell that
contains it. Snow-flagged cells and undefined wetness are excluded. The pooled
station-months are then scored with the same detector metrics used for the gridded
references: the detection contrast (the right diagnostic for a wet versus dry
index), the Spearman and Pearson correlations, and a tercile categorical score.

USCRN is in-situ ground truth from NOAA's own reference network, but it is
contiguous-U.S. soil moisture at climate-reference sites, not inundation, so the
expected result is a modest detection contrast with weak magnitude skill: the
point-scale confirmation that the index is a detector, not a soil-moisture proxy.
"""

import os
import sys

import numpy as np

from swi import validate as val
from swi.io_uscrn import load_station_monthly


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
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    uscrn_dir = sys.argv[1]
    products = sorted(sys.argv[2:])
    stations = load_station_monthly(uscrn_dir)
    if not stations:
        print(f"no USCRN station data found in {uscrn_dir}")
        return 1

    W, S, LAT = [], [], []
    used_stations, used_months = set(), set()
    for p in products:
        lat, lon, wet, snowf, month = read_product(p)
        for wban, st in stations.items():
            sm = st["months"].get(month)
            if sm is None:
                continue
            i = int(np.argmin(np.abs(lat - st["lat"])))
            j = int(np.argmin(np.abs(lon - (st["lon"] % 360.0))))
            wv, sf = wet[i, j], snowf[i, j]
            if not np.isfinite(wv) or wv < 0 or sf > 0.5:
                continue
            W.append(wv); S.append(sm); LAT.append(st["lat"])
            used_stations.add(wban); used_months.add(month)

    W = np.asarray(W); S = np.asarray(S); LAT = np.asarray(LAT)
    s = val.skill_scores(W, S)
    dc = val.detection_contrast(W, S, thr=0.0)
    b_hi = np.quantile(S, 0.667)
    cat = val.categorical(W, S, a_hi=0.0, b_hi=b_hi)
    print(f"\nWET vs USCRN 5 cm soil moisture, point to pixel "
          f"({s['n']:,} station-months, {len(used_stations)} stations, "
          f"{len(used_months)} months):")
    print(f"  Spearman r : {s['spearman_r']:+.3f}   Pearson r {s['pearson_r']:+.3f}")
    print(f"  DETECTION:")
    print(f"    mean SM where WET>0 : {dc['mean_hi']:.3f}  vs  WET=0 : {dc['mean_lo']:.3f}"
          f"   ({dc['ratio']:.2f}x, n_wet={dc['n_hi']:,})")
    print(f"    WET>0 predicts SM>tercile: POD={cat['POD']:.2f} FAR={cat['FAR']:.2f} "
          f"CSI={cat['CSI']:.2f} HSS={cat['HSS']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
