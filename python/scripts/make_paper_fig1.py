"""Regenerable Figure 1 for the reconstruction paper: the three-panel monthly product.

    python -m scripts.make_paper_fig1 MONTHLY_PRODUCT.nc [--pass dsc] [--out F.png]
    python -m scripts.make_paper_fig1 --from-cache CACHE.npz [--out F.png]

Panels: wetness index, near-surface temperature (RTEMP), and snow frequency.
The temperature panel is labelled as a near-surface temperature, matching the
paper's finding that the historical land-skin-temperature label appears wrong.
Older products carry the variable as land_skin_temperature_mean and newer ones
as retrieval_temperature_mean; both are read.

A raw build also writes a small reproducibility cache (the plotted arrays plus
provenance) next to results/figure_cache/, so the figure can be redrawn later
with --from-cache even if the monthly product is gone. See swi.figcache.
"""

import os
import re
import sys

import numpy as np

from swi import figcache


def opt(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def _plot(lat, lon, wet, temp, snow, month, pass_, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    roll = lon.size // 2
    ext = [-180, 180, float(lat.min()), float(lat.max())]
    view = lambda a: np.roll(a, roll, axis=1)
    wet = np.where(wet >= 0, wet, np.nan)
    temp = np.where(temp > -90, temp, np.nan)
    snow = np.where(snow > 0, snow, np.nan)

    # Author at the ~6.5 in text width so titles/labels land >= 8 pt on the page
    # after the figure-to-page reduction (paper-figure-standards.md, section 2).
    w = 6.5; sc = w / 6.5
    tsz, lab, tick, plab = 10 * sc, 10 * sc, 9 * sc, 12 * sc
    fig, axes = plt.subplots(3, 1, figsize=(w, 7.1))
    panels = [
        ("(a)", view(wet), "Wetness index (WET)", "YlGnBu", 0, 100, "index 0-100"),
        ("(b)", view(temp), "Near-surface temperature (RTEMP)", "turbo", 230, 320, "K"),
        ("(c)", view(snow), "Snow frequency", "PuBu", 0, 1, "fraction of days"),
    ]
    for ax, (letter, img, title, cmap, lo, hi, unit) in zip(axes, panels):
        im = ax.imshow(img, origin="lower", extent=ext, aspect="auto",
                       cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(f"{title}, monthly mean, {month} ({pass_})", fontsize=tsz, pad=4)
        ax.tick_params(labelsize=tick)
        cb = fig.colorbar(im, ax=ax, shrink=0.85, label=unit)
        cb.set_label(unit, fontsize=lab); cb.ax.tick_params(labelsize=tick)
        ax.text(0.015, 0.96, letter, transform=ax.transAxes, fontsize=plab,
                fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          alpha=0.75, edgecolor="none"))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    argv = sys.argv
    out = opt(argv, "--out", "../docs/figures/fig_product.png")

    if "--from-cache" in argv:
        arrays, meta = figcache.load(opt(argv, "--from-cache"))
        _plot(arrays["lat"], arrays["lon"], arrays["wet"], arrays["temp"],
              arrays["snow"], str(meta["month"]), str(meta["pass"]), out)
        return 0

    if len(argv) < 2:
        print(__doc__)
        return 1
    path = argv[1]
    pass_ = opt(argv, "--pass", "dsc")
    cache = opt(argv, "--cache", figcache.cache_path("fig_product"))

    import netCDF4 as nc
    ds = nc.Dataset(path)
    try:
        lat = np.asarray(ds["lat"][:], np.float64)
        lon = np.asarray(ds["lon"][:], np.float64)
        wet = np.ma.filled(ds[f"wetness_index_mean_{pass_}"][:], np.nan)
        tvar = (f"retrieval_temperature_mean_{pass_}"
                if f"retrieval_temperature_mean_{pass_}" in ds.variables
                else f"land_skin_temperature_mean_{pass_}")
        temp = np.ma.filled(ds[tvar][:], np.nan)
        snow = np.ma.filled(ds[f"snow_frequency_{pass_}"][:], np.nan)
        # Prefer the file's own month attribute; otherwise take a YYYYMM token
        # from the filename so the panel title stays short (a bare basename
        # overflows the panel).
        month = getattr(ds, "month", None)
        if month is None:
            m = re.search(r"(\d{6})", os.path.basename(path))
            month = m.group(1) if m else os.path.splitext(os.path.basename(path))[0]
        month = str(month)
    finally:
        ds.close()

    # Plot the arrays exactly as the cache stores them (float32), so a
    # --from-cache redraw is identical to this raw build rather than merely close.
    stored = figcache.as_stored(lat=lat, lon=lon, wet=wet, temp=temp, snow=snow)
    _plot(stored["lat"], stored["lon"], stored["wet"], stored["temp"],
          stored["snow"], month, pass_, out)

    # Save the cache only after a successful render, so a failed plot cannot
    # leave a fresh cache beside a stale or missing figure.
    if "--no-cache" not in argv:
        figcache.save(cache, {
            "figure": "fig_product",
            "script": "python/scripts/make_paper_fig1.py",
            "built_from": os.path.basename(path),
            "product_sha256": figcache.sha256(path),
            "pass": pass_,
            "month": month,
        }, **stored)
    return 0


if __name__ == "__main__":
    sys.exit(main())
