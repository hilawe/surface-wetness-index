"""Validate a monthly WET product against SWAMPS fractional surface water.

    python -m scripts.validate_swamps PRODUCT.nc SWAMPS_FW.nc [--png OUT]

SWAMPS (Surface WAter Microwave Product Series) fractional inundation is the
reference that matches what the Basist index actually is: a surface-wetness and
inundation detector. We test whether WET fires where there is more open or
inundated surface water.
"""

import os
import re
import sys

import numpy as np

from swi import figcache, grids, validate as val


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
    if "--from-cache" in sys.argv:
        cpath = sys.argv[sys.argv.index("--from-cache") + 1]
        out = sys.argv[sys.argv.index("--png") + 1] if "--png" in sys.argv else None
        arrays, meta = figcache.load(cpath)
        _map(arrays["lat"], arrays["lon"], arrays["wet"], arrays["fw"],
             arrays["mask"], out, _wet_label(meta.get("built_from")))
        return 0

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

    # Restrict to land. SWAMPS carries fractional water over ocean too, and
    # without this mask the "over land" statistics below were computed over
    # ocean and land together, which inflated the whole-field correlation and
    # the detection contrast against a mostly-water population.
    land = grids.land_mask(plat, plon)
    if land.all():
        # grids.land_mask falls back to all-True when global_land_mask is
        # missing. Silently accepting that would recreate the ocean-plus-land
        # comparison this mask exists to prevent, while the output still says
        # "over land". Fail instead.
        raise RuntimeError(
            "land mask is all-True, so no ocean was excluded. Install "
            "global_land_mask; these statistics are land-only by definition.")
    m = land & np.isfinite(fw_on) & (wet >= 0) & ~(snowf > 0.5)
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
        # Plot the float32-stored arrays so a --from-cache redraw is identical,
        # and save the cache only after a successful render.
        stored = figcache.as_stored(lat=plat, lon=plon, wet=wet, fw=fw_on, mask=m)
        _map(stored["lat"], stored["lon"], stored["wet"], stored["fw"],
             stored["mask"], png, _wet_label(product))
        if "--no-cache" not in sys.argv:
            figcache.save(figcache.cache_path("fig_swamps"), {
                "figure": "fig_swamps",
                "script": "python/scripts/validate_swamps.py",
                "built_from": os.path.basename(product),
                "product_sha256": figcache.sha256(product),
                "swamps_file": os.path.basename(swamps),
                "swamps_sha256": figcache.sha256(swamps),
            }, **stored)
    return 0


def _wet_label(product_name):
    """Panel (a) title, in the same form the other validators use.

    Figure 3 labeled this panel "SWI WET" while the ESA CCI, SMAP and ERA5
    figures labeled the same quantity "SWI WET (F-16, YYYYMM)". A reader
    meeting both in adjacent figures has to work out they are the same thing,
    so the sensor and month are carried here too, derived from the product the
    figure was built from rather than hardcoded, because this script also draws
    the F-13 1998 comparison.
    """
    m = re.match(r"SWI_(F\d+)_(\d{6})_", os.path.basename(product_name or ""))
    if not m:
        return "SWI WET"
    sat, ym = m.group(1), m.group(2)
    return f"SWI WET ({sat[0]}-{sat[1:]}, {ym})"


def _map(lat, lon, wet, fw, mask, out, wet_label="SWI WET"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Author at the ~6.5 in text width so titles/labels land >= 8 pt on the page
    # after the figure-to-page reduction (paper-figure-standards.md, section 2).
    w = 6.5; sc = w / 6.5
    tsz, tick, plab = 10 * sc, 9 * sc, 12 * sc
    roll = lon.size // 2; ext = [-180, 180, lat.min(), lat.max()]
    v = lambda a: np.roll(np.where(mask, a, np.nan), roll, axis=1)
    fig, ax = plt.subplots(2, 1, figsize=(w, 4.8))
    for a, letter, (img, ttl, cm, lo, hi) in zip(ax, ("(a)", "(b)"), [
            (v(wet), wet_label, "YlGnBu", 0, 100),
            (v(fw), "SWAMPS fractional surface water", "Blues", 0, 0.5)]):
        im = a.imshow(img, origin="lower", extent=ext, aspect="auto", cmap=cm,
                      vmin=lo, vmax=hi)
        a.set_title(ttl, fontsize=tsz, pad=4)
        a.tick_params(labelsize=tick)
        cb = fig.colorbar(im, ax=a, shrink=0.85); cb.ax.tick_params(labelsize=tick)
        a.text(0.015, 0.96, letter, transform=a.transAxes, fontsize=plab,
               fontweight="bold", va="top", ha="left",
               bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                         alpha=0.75, edgecolor="none"))
    fig.suptitle("Surface Wetness Index vs SWAMPS inundation (co-located land)",
                 fontsize=tsz * 1.1)
    fig.tight_layout(); fig.savefig(out, dpi=150); print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
