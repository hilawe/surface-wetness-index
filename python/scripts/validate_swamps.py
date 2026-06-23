"""Validate a monthly WET product against SWAMPS fractional surface water.

    python -m scripts.validate_swamps PRODUCT.nc SWAMPS_FW.nc [--png OUT]

SWAMPS (Surface WAter Microwave Product Series) fractional inundation is the
reference that matches what the Basist index actually is: a surface-wetness and
inundation detector. We test whether WET fires where there is more open or
inundated surface water.
"""

import os
import sys

import numpy as np

from swi import validate as val


def load_swamps_fw(path):
    """Return (lat_asc, lon_0360, fw) for SWAMPS fractional water, our orientation."""
    import netCDF4 as nc
    ds = nc.Dataset(path)
    try:
        lat = np.asarray(ds["lat"][:], np.float64)
        lon = np.asarray(ds["lon"][:], np.float64)
        fw = np.squeeze(np.ma.filled(ds["fw"][:], np.nan)).astype(np.float64)
    finally:
        ds.close()
    if lat[0] > lat[-1]:
        lat = lat[::-1]; fw = fw[::-1, :]
    lon = np.where(lon < 0, lon + 360.0, lon)
    order = np.argsort(lon)
    return lat, lon[order], fw[:, order]


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
    product, swamps = sys.argv[1], sys.argv[2]
    png = None
    if "--png" in sys.argv:
        png = sys.argv[sys.argv.index("--png") + 1]

    plat, plon, wet, snowf = read_product(product)
    slat, slon, fw = load_swamps_fw(swamps)
    fw_on = val.regrid_nearest(slat, slon, fw, plat, plon)

    m = np.isfinite(fw_on) & (wet >= 0) & ~(snowf > 0.5)
    latm = np.broadcast_to(plat[:, None], wet.shape)[m]
    W, F = wet[m], fw_on[m]
    dc = val.detection_contrast(W, F, thr=0.0)
    # treat "inundated" as fw above a small open-water fraction
    f_hi = 0.05
    cat = val.categorical(W, F, a_hi=0.0, b_hi=f_hi)
    s = val.skill_scores(W, F)
    print(f"\nWET vs SWAMPS fractional surface water, over land (n={s['n']:,}):")
    print(f"  whole-field Spearman r : {s['spearman_r']:+.3f}   Pearson r {s['pearson_r']:+.3f}")
    print(f"  DETECTION:")
    print(f"    mean fw where WET>0 : {dc['mean_hi']:.4f}  vs  WET=0 : {dc['mean_lo']:.4f}"
          f"   ({dc['ratio']:.2f}x, n_wet={dc['n_hi']:,})")
    print(f"    WET>0 predicts fw>{f_hi}: POD={cat['POD']:.2f} FAR={cat['FAR']:.2f} "
          f"CSI={cat['CSI']:.2f} HSS={cat['HSS']:.2f}")
    # conditional: among high-inundation cells, how often does WET fire?
    hi = F > 0.10
    if hi.sum() > 100:
        print(f"    where fw>0.10 (clearly inundated): WET>0 in {100*(W[hi]>0).mean():.0f}% "
              f"of {int(hi.sum()):,} cells")
    print(f"  DETECTION by latitude zone (contrast = mean fw at WET>0 / WET=0):")
    for name, z in val.detection_by_zone(W, F, latm, thr=0.0).items():
        print(f"    {name:<14}: {z['ratio']:.2f}x  (mean fw {z['mean_hi']:.4f} vs "
              f"{z['mean_lo']:.4f}, n_wet={z['n_hi']:,})")

    if png:
        _map(plat, plon, wet, fw_on, m, png)
    return 0


def _map(lat, lon, wet, fw, mask, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    roll = lon.size // 2; ext = [-180, 180, lat.min(), lat.max()]
    v = lambda a: np.roll(np.where(mask, a, np.nan), roll, axis=1)
    fig, ax = plt.subplots(2, 1, figsize=(11, 8))
    for a, (img, t, cm, lo, hi) in zip(ax, [
            (v(wet), "SWI WET", "YlGnBu", 0, 100),
            (v(fw), "SWAMPS fractional surface water", "Blues", 0, 0.5)]):
        im = a.imshow(img, origin="lower", extent=ext, aspect="auto", cmap=cm,
                      vmin=lo, vmax=hi)
        a.set_title(t); fig.colorbar(im, ax=a, shrink=0.85)
    fig.suptitle("Surface Wetness Index vs SWAMPS inundation (co-located land)")
    fig.tight_layout(); fig.savefig(out, dpi=110); print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
