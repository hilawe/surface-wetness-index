"""Regenerable Figure 1 for the revival paper: the three-panel monthly product.

    python -m scripts.make_paper_fig1 MONTHLY_PRODUCT.nc [--pass dsc] [--out F.png]

Panels: wetness index, near-surface temperature (RTEMP), and snow frequency.
The temperature panel is labelled as a near-surface temperature, matching the
paper's finding that the historical land-skin-temperature label appears wrong.
Older products carry the variable as land_skin_temperature_mean and newer ones
as retrieval_temperature_mean; both are read.
"""

import os
import sys

import numpy as np


def opt(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def main():
    argv = sys.argv
    if len(argv) < 2:
        print(__doc__)
        return 1
    path = argv[1]
    pass_ = opt(argv, "--pass", "dsc")
    out = opt(argv, "--out", "../docs/figures/fig_product.png")

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
        month = str(getattr(ds, "month", os.path.basename(path)))
    finally:
        ds.close()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    roll = lon.size // 2
    ext = [-180, 180, lat.min(), lat.max()]
    view = lambda a: np.roll(a, roll, axis=1)
    wet = np.where(wet >= 0, wet, np.nan)
    temp = np.where(temp > -90, temp, np.nan)
    snow = np.where(snow > 0, snow, np.nan)

    fig, axes = plt.subplots(3, 1, figsize=(11, 12))
    panels = [
        (view(wet), "Wetness index (WET)", "YlGnBu", 0, 100, "index 0-100"),
        (view(temp), "Near-surface temperature (RTEMP)", "turbo", 230, 320, "K"),
        (view(snow), "Snow frequency", "PuBu", 0, 1, "fraction of days"),
    ]
    for ax, (img, title, cmap, lo, hi, unit) in zip(axes, panels):
        im = ax.imshow(img, origin="lower", extent=ext, aspect="auto",
                       cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(f"{title}, monthly mean, {month} ({pass_})")
        fig.colorbar(im, ax=ax, shrink=0.85, label=unit)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
