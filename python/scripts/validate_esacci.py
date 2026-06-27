"""Validate a monthly WET product against ESA CCI microwave soil moisture.

    python -m scripts.validate_esacci PRODUCT.nc YYYYMM [--data-dir DIR] [--png OUT]

ESA CCI Soil Moisture (COMBINED, v09.2) is the community microwave soil-moisture
climate data record. It is observation-based, but it merges passive-microwave
retrievals from many of the same imager channels the wetness index uses, so it
shares the passive-microwave physics family with the index. Shared sensors and
seasonal microwave artifacts can inflate apparent agreement, so ESA CCI is
reported here as a related microwave benchmark rather than as independent
ground truth. It is shown separately from the physically independent
references (USCRN in-situ, ERA5-Land reanalysis, SWAMPS inundation).

Daily CCI files for the month are downloaded from CEDA (open access, no API),
composited to a per-cell monthly mean over flag-clean cells (snow-free,
non-densely-vegetated land), regridded to our grid, and co-located over
common-valid unfrozen land. Spearman rank correlation is the headline (WET is an
index), with the detection contrast as the right diagnostic for a wet/dry index.
"""

import os
import sys

import numpy as np

from swi import validate as val
from swi.io_esacci import PRODUCT, VERSION, load_cci_monthly


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


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    product, month = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    png = None
    data_dir = f"../data/esacci/{month}"
    if "--png" in rest:
        i = rest.index("--png"); png = rest[i + 1]; rest = rest[:i] + rest[i + 2:]
    if "--data-dir" in rest:
        i = rest.index("--data-dir"); data_dir = rest[i + 1]

    plat, plon, wet, snowf = read_product(product)
    print(f"downloading/compositing ESA CCI {PRODUCT} {VERSION} for {month} ...")
    elat, elon, sm, ndays = load_cci_monthly(month, data_dir)
    sm_on = val.regrid_nearest(elat, elon, sm, plat, plon)

    # common-valid land (CCI is land-only -> NaN over ocean), unfrozen, real wetness
    land = np.isfinite(sm_on)
    unfrozen = ~(snowf > 0.5)
    m = val.common_valid(wet, sm_on, land) & unfrozen & (wet >= 0)

    latm = np.broadcast_to(plat[:, None], wet.shape)[m]
    W, S = wet[m], sm_on[m]
    s = val.skill_scores(W, S)
    pc = val.pattern_correlation(wet, sm_on, m)
    dc = val.detection_contrast(W, S, thr=0.0)
    b_hi = np.quantile(S, 0.667)
    cat = val.categorical(W, S, a_hi=0.0, b_hi=b_hi)
    among = val.skill_scores(W[W > 0], S[W > 0])
    cov = int(np.nansum(ndays > 0))
    print(f"\nWET vs ESA CCI {PRODUCT} soil moisture, {month}, descending, "
          f"over land (n={s['n']:,}):")
    print(f"  CCI monthly coverage   : {cov:,} cells with >=1 clean day")
    print(f"  whole-field Spearman r : {s['spearman_r']:+.3f}   Pearson r {s['pearson_r']:+.3f}")
    print(f"  pattern correlation    : {pc:+.3f}")
    print(f"  DETECTION (the right framing for a wet/dry index):")
    print(f"    mean SM where WET>0  : {dc['mean_hi']:.3f}  vs  WET=0 : {dc['mean_lo']:.3f}"
          f"   ({dc['ratio']:.2f}x, n_wet={dc['n_hi']:,})")
    print(f"    Spearman among WET>0 : {among['spearman_r']:+.3f}  "
          f"(magnitude vs SM magnitude)")
    print(f"    WET>0 predicts SM>tercile: POD={cat['POD']:.2f} FAR={cat['FAR']:.2f} "
          f"CSI={cat['CSI']:.2f} HSS={cat['HSS']:.2f}")
    print(f"  DETECTION by latitude zone (contrast = mean SM at WET>0 / WET=0):")
    for name, z in val.detection_by_zone(W, S, latm, thr=0.0).items():
        print(f"    {name:<14}: {z['ratio']:.2f}x  (mean SM {z['mean_hi']:.3f} vs "
              f"{z['mean_lo']:.3f}, n_wet={z['n_hi']:,})")

    if png:
        _map(plat, plon, wet, sm_on, m, month, png)
    return 0


def _map(lat, lon, wet, sm, mask, month, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    roll = lon.size // 2; ext = [-180, 180, lat.min(), lat.max()]
    v = lambda a: np.roll(np.where(mask, a, np.nan), roll, axis=1)
    fig, ax = plt.subplots(2, 1, figsize=(11, 8))
    for a, (img, t, cm, lo, hi) in zip(ax, [
            (v(wet), f"SWI WET (F-16, {month})", "YlGnBu", 0, 100),
            (v(sm), f"ESA CCI {PRODUCT} soil moisture", "YlGnBu", 0, 0.5)]):
        im = a.imshow(img, origin="lower", extent=ext, aspect="auto", cmap=cm,
                      vmin=lo, vmax=hi)
        a.set_title(t); fig.colorbar(im, ax=a, shrink=0.85)
    fig.suptitle("Surface Wetness Index vs ESA CCI microwave soil moisture "
                 "(co-located land)")
    fig.tight_layout(); fig.savefig(out, dpi=110); print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
