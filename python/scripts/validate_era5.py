"""Validate a monthly WET product against ERA5-Land surface soil moisture.

    python -m scripts.validate_era5 PRODUCT.nc YYYYMM [ERA5_FILE.nc] [--png OUT]

Loads ERA5-Land monthly mean volumetric soil water layer 1 (0 to 7 cm, the layer
the microwave wetness senses), regrids it to our 0.25 degree grid, co-locates
over common-valid land excluding frozen/snow cells, and reports skill (Spearman
rank correlation is the headline, since WET is an index). If ERA5_FILE is not
given and not cached, it is downloaded via the Copernicus CDS API (needs
~/.cdsapirc).
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
    finally:
        ds.close()
    return lat, lon, wet, snowf


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    product, month = sys.argv[1], sys.argv[2]
    era5_path = None
    png = None
    rest = sys.argv[3:]
    if "--png" in rest:
        i = rest.index("--png"); png = rest[i + 1]; rest = rest[:i] + rest[i + 2:]
    if rest:
        era5_path = rest[0]
    if era5_path is None:
        era5_path = f"../data/era5/era5land_swvl1_{month}.nc"
        os.makedirs(os.path.dirname(era5_path), exist_ok=True)

    plat, plon, wet, snowf = read_product(product)
    if not os.path.exists(era5_path):
        print(f"downloading ERA5-Land soil moisture for {month} via CDS ...")
    elat, elon, sm = load_swvl1(month, era5_path)
    sm_on = val.regrid_nearest(elat, elon, sm, plat, plon)

    # common-valid land (ERA5 land-only -> NaN over ocean), unfrozen, real wetness
    land = np.isfinite(sm_on)
    unfrozen = ~(snowf > 0.5)                   # exclude mostly-snow cells
    m = val.common_valid(wet, sm_on, land) & unfrozen & (wet >= 0)

    W, S = wet[m], sm_on[m]
    s = val.skill_scores(W, S)
    pc = val.pattern_correlation(wet, sm_on, m)
    dc = val.detection_contrast(W, S, thr=0.0)
    b_hi = np.quantile(S, 0.667)
    cat = val.categorical(W, S, a_hi=0.0, b_hi=b_hi)
    print(f"\nWET vs ERA5-Land swvl1, {month}, descending, over land (n={s['n']:,}):")
    print(f"  whole-field Spearman r : {s['spearman_r']:+.3f}   Pearson r {s['pearson_r']:+.3f}")
    print(f"  pattern correlation    : {pc:+.3f}")
    print(f"  DETECTION (the right framing for a wet/dry index):")
    print(f"    mean ref where WET>0 : {dc['mean_hi']:.3f}  vs  WET=0 : {dc['mean_lo']:.3f}"
          f"   ({dc['ratio']:.2f}x, n_wet={dc['n_hi']:,})")
    print(f"    WET>0 predicts SM>tercile: POD={cat['POD']:.2f} FAR={cat['FAR']:.2f} "
          f"CSI={cat['CSI']:.2f} HSS={cat['HSS']:.2f}")

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
            (v(sm), "ERA5-Land soil water layer 1", "YlGnBu", 0, 0.5)]):
        im = a.imshow(img, origin="lower", extent=ext, aspect="auto", cmap=cm,
                      vmin=lo, vmax=hi)
        a.set_title(t); fig.colorbar(im, ax=a, shrink=0.85)
    fig.suptitle("Surface Wetness Index vs ERA5-Land soil moisture (co-located land)")
    fig.tight_layout(); fig.savefig(out, dpi=110); print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
