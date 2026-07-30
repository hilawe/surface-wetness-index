"""Two-panel density scatter: the index against surface water and soil moisture.

    python -m scripts.make_fig_scatter SWAMPS_PRODUCT.nc SWAMPS_FW.nc \
        CCI_PRODUCT.nc CCI_MONTH [--out F.png]

Panel (a) co-locates a monthly product with SWAMPS fractional surface water and
panel (b) a monthly product with the ESA CCI soil-moisture composite, using the
same masking as the corresponding validation drivers, so the correlations the
panels illustrate are the committed ones. The cell populations are drawn from
the data files; the annotated correlation values are read live but should match
the committed artifacts, and the script prints them for that check.
"""

import os
import sys

import numpy as np

from swi import grids, validate as val
from swi.io_esacci import load_cci_monthly


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
    finally:
        ds.close()
    return lat, lon, wet, snowf


def main():
    argv = sys.argv
    if len(argv) < 5:
        print(__doc__)
        return 1
    swamps_product, swamps_fw = argv[1], argv[2]
    cci_product, cci_month = argv[3], argv[4]
    out = opt(argv, "--out", "../docs/figures/fig_scatter.png")

    from scripts.validate_swamps import load_swamps_fw

    # Panel (a): same population as swamps_transfer_test.
    plat, plon, wet_a, snowf_a = read_product(swamps_product)
    slat, slon, fw = load_swamps_fw(swamps_fw)
    fw_on = val.regrid_nearest(slat, slon, fw, plat, plon)
    land = grids.land_mask(plat, plon)
    m_a = land & np.isfinite(fw_on) & (wet_a >= 0) & ~(snowf_a > 0.5)
    Wa, Fa = wet_a[m_a], fw_on[m_a]
    sa = val.skill_scores(Wa, Fa)
    print(f"(a) n={sa['n']:,}  Pearson {sa['pearson_r']:+.3f}  "
          f"Spearman {sa['spearman_r']:+.3f}")

    # Panel (b): same population as validate_esacci.
    plat2, plon2, wet_b, snowf_b = read_product(cci_product)
    elat, elon, sm, _ = load_cci_monthly(cci_month, f"../data/esacci/{cci_month}")
    sm_on = val.regrid_nearest(elat, elon, sm, plat2, plon2)
    m_b = (val.common_valid(wet_b, sm_on, np.isfinite(sm_on))
           & ~(snowf_b > 0.5) & (wet_b >= 0))
    Wb, Sb = wet_b[m_b], sm_on[m_b]
    sb = val.skill_scores(Wb, Sb)
    print(f"(b) n={sb['n']:,}  Pearson {sb['pearson_r']:+.3f}  "
          f"Spearman {sb['spearman_r']:+.3f}")

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    w = 8.5
    sc = w / 6.5
    t, lab, tick = 10 * sc, 10 * sc, 9 * sc
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(w, 4.0),
                                   constrained_layout=True)

    h1 = ax1.hist2d(Fa, Wa, bins=[60, 60], range=[[0, 0.6], [0, 60]],
                    norm=LogNorm(), cmap="viridis")
    fig.colorbar(h1[3], ax=ax1, label="cells", shrink=0.9)
    ax1.set_xlabel("SWAMPS fractional water, July 2016", fontsize=lab)
    ax1.set_ylabel("monthly wetness index (WET)", fontsize=lab)
    ax1.set_title("Against surface water", fontsize=t, pad=4)
    ax1.text(0.97, 0.96,
             f"Pearson {sa['pearson_r']:+.2f}\nSpearman {sa['spearman_r']:+.2f}",
             transform=ax1.transAxes, fontsize=tick, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       alpha=0.85, edgecolor="0.7"))
    ax1.tick_params(labelsize=tick)
    ax1.text(0.015, 0.96, "(a)", transform=ax1.transAxes, fontsize=12 * sc,
             fontweight="bold", va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                       alpha=0.75, edgecolor="none"))

    h2 = ax2.hist2d(Sb, Wb, bins=[60, 60], range=[[0, 0.6], [0, 60]],
                    norm=LogNorm(), cmap="viridis")
    fig.colorbar(h2[3], ax=ax2, label="cells", shrink=0.9)
    ax2.set_xlabel("ESA CCI soil moisture, July 2023", fontsize=lab)
    ax2.set_title("Against soil moisture", fontsize=t, pad=4)
    ax2.text(0.97, 0.96,
             f"Pearson {sb['pearson_r']:+.2f}\nSpearman {sb['spearman_r']:+.2f}",
             transform=ax2.transAxes, fontsize=tick, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       alpha=0.85, edgecolor="0.7"))
    ax2.tick_params(labelsize=tick)
    ax2.text(0.015, 0.96, "(b)", transform=ax2.transAxes, fontsize=12 * sc,
             fontweight="bold", va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                       alpha=0.75, edgecolor="none"))

    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
