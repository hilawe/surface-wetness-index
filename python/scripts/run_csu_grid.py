"""Run the Surface Wetness Index engine on a CSU ICDR-GRID daily file.

Usage:
    python -m scripts.run_csu_grid /path/to/CSU_*_ICDR-GRID_*.nc [asc|dsc] [out.png]

Prints summary statistics and, if an output path is given, writes a quick-look
map of WET, RTEMP, and SNOW. Intended as an end-to-end demonstration of the
read -> engine path on real gridded brightness temperatures.
"""

import sys

import numpy as np

from swi import io_csu_grid as io


def summarize(r):
    v, wet, snow, temp = r["valid"], r["wet"], r["snow"], r["temp"]
    print(f"sensor={r['sensor']}  pass={r['pass']}")
    if r.get("empty_channels"):
        print(f"  WARNING: required channel(s) entirely missing: "
              f"{', '.join(r['empty_channels'])}")
        print("  -> the algorithm needs all seven channels; no product for this "
              "pass. (F-17 37V has been inoperable since 2016; use F-16 or a "
              "pre-2016 F-17 date.)")
    print(f"  valid input cells : {v.sum():,} / {v.size:,} ({100*v.sum()/v.size:.1f}%)")
    print(f"  WET>0 wet surface : {(wet>0).sum():,}  (max {np.nanmax(wet):.1f})")
    print(f"  SNOW>0 scattering : {(snow>0).sum():,}")
    print(f"  SNOW=-1 ice       : {(snow==-1).sum():,}")
    print(f"  valid RTEMP       : {(temp>-90).sum():,}")


def make_map(r, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon, lat = r["lon"], r["lat"]
    # roll 0..360 grid to a -180..180 view for a conventional map
    roll = lon.size // 2
    ext = [-180, 180, lat.min(), lat.max()]

    def view(a):
        return np.roll(a, roll, axis=1)

    wet = np.where(r["wet"] >= 0, r["wet"], np.nan)
    temp = np.where(r["temp"] > -90, r["temp"], np.nan)
    snow = np.where(r["snow"] > 0, r["snow"], np.nan)

    fig, axes = plt.subplots(3, 1, figsize=(11, 12))
    panels = [
        (view(wet), "Surface Wetness Index (WET)", "YlGnBu", 0, 100, "index 0-100"),
        (view(temp), "Land Skin Temperature (RTEMP)", "turbo", 230, 320, "K"),
        (view(snow), "Snow scattering (SNOW > 0)", "BuPu", 0, 40, "scattering"),
    ]
    for ax, (img, title, cmap, vmin, vmax, cb) in zip(axes, panels):
        im = ax.imshow(img, origin="lower", extent=ext, aspect="auto",
                       cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        fig.colorbar(im, ax=ax, shrink=0.85, label=cb)

    caveats = {
        "amsr2": "AMSR2 channels approximate SSM/I (18.7/23.8/36.5/89 vs "
                 "19/22/37/85); data-path proof of concept, not validated.",
        "ssmis": "SSMIS 91 GHz substitutes for 85 GHz (coefficients tuned on "
                 "85); 85-to-91 calibration pending. Research output, not yet validated.",
        "ssmi":  "Native SSM/I channels. Research output, not yet validated.",
    }
    fig.suptitle(
        f"Basist Surface Wetness Index engine on CSU ICDR-GRID  "
        f"({r['sensor'].upper()} {r['pass']} pass)\n"
        f"{caveats.get(r['sensor'], 'Research output, not yet validated.')}",
        fontsize=12)
    fig.subplots_adjust(top=0.90, hspace=0.30)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = sys.argv[1]
    pass_ = sys.argv[2] if len(sys.argv) > 2 else "dsc"
    out = sys.argv[3] if len(sys.argv) > 3 else None
    with np.errstate(divide="ignore", invalid="ignore"):
        r = io.evaluate_file(path, pass_=pass_)
    summarize(r)
    if out:
        make_map(r, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
