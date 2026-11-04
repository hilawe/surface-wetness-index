"""Validate a monthly WET product against SMAP L-band soil moisture.

    python -m scripts.validate_smap PRODUCT.nc YYYYMM [--data-dir DIR] [--png OUT]
                                    [--offline]

--offline skips every network call and uses only the cached daily files, so a
result can be regenerated when the archive metadata service is unreachable.

SMAP (SPL3SMP_E v006) retrieves soil moisture at L-band, 1.4 GHz. It shares no
channel, no sensor, and no seasonal artifact family with the 19 to 85 GHz imager
channels the wetness index uses, which makes it a PHYSICALLY INDEPENDENT
reference. ESA CCI COMBINED is not, because it merges passive retrievals from
many of the same DMSP channels as the index.

A prediction recorded in docs/SMAP_Comparison_Scope.md before any number existed
held that agreement with SMAP would be LOWER than the 1.43 measured against ESA
CCI. It was not confirmed. Report that ordering and stop there. The result does
not identify the direction or magnitude of shared-channel bias in the ESA CCI
comparison, and it does not establish that reference quality caused the
difference.

The AM (descending, about 6 am local) retrieval is used, matched to the wetness
index descending pass. SMAP arrives on a 9 km grid and is area-binned to the
0.25 degree product grid inside the loader, so no further regridding happens
here. Statistics mirror validate_esacci.py exactly so the two rows of the
manuscript table are comparable rather than merely adjacent.

The quality screen is not free, and the DOMAIN CHECK below runs before any skill
statistic to say what it costs. It separates two things an earlier version
conflated. SPATIAL SUPPORT is where SMAP retrieves at all, which is mostly
geography. QUALITY RETENTION is the fraction of those retrievals surviving the
recommended screen, which is the only part that is a property of the screen.
Which surface types the screen removes is NOT measured here, so no claim is made
about it.
"""

import os
import sys

import numpy as np

from swi import figcache
from swi import validate as val
from swi.io_smap import SHORT_NAME, VERSION, load_smap_monthly


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
        _map(arrays["lat"], arrays["lon"], arrays["wet"], arrays["sm"],
             arrays["mask"], str(meta["month"]), out)
        return 0

    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    product, month = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    png = json_out = None
    data_dir = "../data/smap"
    if "--png" in rest:
        i = rest.index("--png"); png = rest[i + 1]; rest = rest[:i] + rest[i + 2:]
    if "--json" in rest:
        i = rest.index("--json"); json_out = rest[i + 1]; rest = rest[:i] + rest[i + 2:]
    if "--data-dir" in rest:
        i = rest.index("--data-dir"); data_dir = rest[i + 1]

    plat, plon, wet, snowf = read_product(product, "dsc")
    print(f"compositing SMAP {SHORT_NAME} {VERSION} AM for {month} ...")
    dl = "--offline" not in sys.argv
    sm_on, nday, st = load_smap_monthly(month, data_dir, plat, plon, download=dl)
    # the same month on the relaxed screen, needed as the denominator for the
    # quality diagnostic below and as the sensitivity population
    base_on, _, _ = load_smap_monthly(month, data_dir, plat, plon, download=False,
                                      screen="successful", verbose=False)

    print(f"  days used              : {st['n_days_used']}")
    print(f"  9 km successful retrievals: {st['fine_successful_retrievals']:,}")
    print(f"  after recommended screen  : {st['fine_kept_by_screen']:,} "
          f"({100 * st['kept_fraction_of_successful']:.1f}% of successful)")
    print(f"  0.25 deg cells with data: {st['target_cells_with_data']:,}")

    land = np.isfinite(sm_on)
    unfrozen = ~(snowf > 0.5)
    m = val.common_valid(wet, sm_on, land) & unfrozen & (wet >= 0)

    # ---- DOMAIN CHECK, before any skill number ----------------------------
    # Two separate questions, and an earlier version conflated them into one
    # misleading ratio. SPATIAL SUPPORT asks where SMAP retrieves at all, which
    # is mostly geography (ocean, ice, no overpass). QUALITY RETENTION asks, of
    # the cells SMAP did retrieve, how many survive the recommended screen. Only
    # the second is a property of the screen, so only the second may be described
    # as one.
    have_wet = np.isfinite(wet) & (wet >= 0) & unfrozen
    fires = have_wet & (wet > 0)
    quiet = have_wet & (wet == 0)
    base = np.isfinite(base_on)

    sup_f = float(np.sum(fires & base)) / max(int(np.sum(fires)), 1)
    sup_q = float(np.sum(quiet & base)) / max(int(np.sum(quiet)), 1)
    qf_den, qq_den = int(np.sum(fires & base)), int(np.sum(quiet & base))
    qf_num, qq_num = int(np.sum(fires & land)), int(np.sum(quiet & land))
    keep_f = qf_num / max(qf_den, 1)
    keep_q = qq_num / max(qq_den, 1)
    ratio = keep_f / keep_q if keep_q > 0 else float("nan")

    print("\n  DOMAIN CHECK")
    print(f"    SMAP spatial support   : firing {100 * sup_f:5.1f}%   "
          f"quiet {100 * sup_q:5.1f}%   (where SMAP retrieves at all)")
    print(f"    quality retention      : firing {100 * keep_f:5.1f}% "
          f"({qf_num:,} of {qf_den:,})   quiet {100 * keep_q:5.1f}% "
          f"({qq_num:,} of {qq_den:,})")
    if ratio < 0.9:
        verdict = "the screen removes firing cells more often, read the skill with care"
    elif ratio <= 1.1:
        verdict = "the screen treats firing and quiet cells alike"
    else:
        verdict = "the screen removes quiet cells more often"
    print(f"    quality-only ratio     : {ratio:.3f}   ({verdict})")
    print("    NOTE: which surface types the screen removes is not measured here.")

    latm = np.broadcast_to(plat[:, None], wet.shape)[m]
    lonm = np.broadcast_to(plon[None, :], wet.shape)[m]
    W, S = wet[m], sm_on[m]
    s = val.skill_scores(W, S)
    pc = val.pattern_correlation(wet, sm_on, m)
    dc = val.detection_contrast(W, S, thr=0.0)
    b_hi = np.quantile(S, 0.667)
    cat = val.categorical(W, S, a_hi=0.0, b_hi=b_hi)
    among = val.skill_scores(W[W > 0], S[W > 0])

    print(f"\nWET vs SMAP {SHORT_NAME} {VERSION} AM soil moisture, {month}, descending, "
          f"over land (n={s['n']:,}):")
    print(f"  whole-field Spearman r : {s['spearman_r']:+.3f}   Pearson r {s['pearson_r']:+.3f}")
    print(f"  pattern correlation    : {pc:+.3f}")
    print("  DETECTION (the right framing for a wet/dry index):")
    print(f"    mean SM where WET>0  : {dc['mean_hi']:.3f}  vs  WET=0 : {dc['mean_lo']:.3f}"
          f"   ({dc['ratio']:.2f}x, n_wet={dc['n_hi']:,})")
    print(f"    Spearman among WET>0 : {among['spearman_r']:+.3f}")
    print(f"    WET>0 predicts SM>tercile: POD={cat['POD']:.2f} FAR={cat['FAR']:.2f} "
          f"CSI={cat['CSI']:.2f} HSS={cat['HSS']:.2f}")
    print("  DETECTION by latitude zone (contrast = mean SM at WET>0 / WET=0):")
    for name, z in val.detection_by_zone(W, S, latm, thr=0.0).items():
        print(f"    {name:<14}: {z['ratio']:.2f}x  (mean SM {z['mean_hi']:.3f} vs "
              f"{z['mean_lo']:.3f}, n_wet={z['n_hi']:,})")

    def contrast_ratio(idx):
        return val.detection_contrast(W[idx], S[idx], thr=0.0)["ratio"]
    ci = val.block_bootstrap_ci(contrast_ratio,
                                val.block_ids(latm, lonm, 10.0),
                                n_draws=2000, seed=20260731, null_value=1.0)
    print(f"    contrast 95% interval (10-degree blocks): "
          f"[{ci['lo']:.2f}, {ci['hi']:.2f}] over {ci['n_blocks']} blocks")
    print(f"\n  PRE-REGISTERED CHECK. Expected below the 1.43 measured against ESA "
          f"CCI. Observed {dc['ratio']:.2f}, interval [{ci['lo']:.2f}, {ci['hi']:.2f}]. "
          f"The expectation was not confirmed. See smap_cci_matched for the paired test.")

    if json_out:
        import json
        res = {
            "product": os.path.basename(product),
            "reference": f"SMAP {SHORT_NAME} {VERSION} AM (L-band)",
            "month": month,
            "n_cells": int(s["n"]),
            "ingest": {k: (v if isinstance(v, str) else
                           float(v) if k == "kept_fraction_of_successful" else int(v))
                       for k, v in st.items()},
            "domain_check": {
                "note": "spatial support and quality retention are separate; only "
                        "the second is a property of the screen",
                "spatial_support_firing": float(sup_f),
                "spatial_support_quiet": float(sup_q),
                "quality_firing_kept": qf_num,
                "quality_firing_available": qf_den,
                "quality_quiet_kept": qq_num,
                "quality_quiet_available": qq_den,
                "quality_retention_firing": float(keep_f),
                "quality_retention_quiet": float(keep_q),
                "quality_retention_ratio": float(ratio),
            },
            "spearman": float(s["spearman_r"]),
            "pearson": float(s["pearson_r"]),
            "pattern_correlation": float(pc),
            "detection_contrast": {k: (int(v) if k.startswith("n_") else float(v))
                                   for k, v in dc.items()},
            "contrast_bootstrap_spatial_blocks": {
                "block_deg": 10.0, "seed": 20260731,
                "lo": float(ci["lo"]), "hi": float(ci["hi"]),
                "n_blocks": int(ci["n_blocks"]), "n_draws": int(ci["n_draws"])},
            "spearman_among_firing": float(among["spearman_r"]),
            "categorical_vs_sm_tercile": {k: float(v) for k, v in cat.items()},
            "contrast_by_zone": {name: {k: (int(v) if k.startswith("n_") else float(v))
                                        for k, v in z.items()}
                                 for name, z in
                                 val.detection_by_zone(W, S, latm, thr=0.0).items()},
        }
        os.makedirs(os.path.dirname(json_out) or ".", exist_ok=True)
        with open(json_out, "w") as f:
            json.dump(res, f, indent=2, sort_keys=True)
        print(f"wrote {json_out}")

    if png:
        stored = figcache.as_stored(lat=plat, lon=plon, wet=wet, sm=sm_on, mask=m)
        _map(stored["lat"], stored["lon"], stored["wet"], stored["sm"],
             stored["mask"], month, png)
        if "--no-cache" not in sys.argv:
            figcache.save(figcache.cache_path("fig_smap"), {
                "figure": "fig_smap",
                "script": "python/scripts/validate_smap.py",
                "built_from": os.path.basename(product),
                "product_sha256": figcache.sha256(product),
                "reference": f"SMAP {SHORT_NAME} {VERSION} AM",
                "month": month,
            }, **stored)
    return 0


def _map(lat, lon, wet, sm, mask, month, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = 6.5; sc = w / 6.5
    tsz, tick, plab = 10 * sc, 9 * sc, 12 * sc
    roll = lon.size // 2; ext = [-180, 180, lat.min(), lat.max()]
    v = lambda a: np.roll(np.where(mask, a, np.nan), roll, axis=1)
    fig, ax = plt.subplots(2, 1, figsize=(w, 4.8))
    for a, letter, (img, ttl, cm, lo, hi) in zip(ax, ("(a)", "(b)"), [
            (v(wet), f"SWI WET (F-16, {month})", "YlGnBu", 0, 100),
            (v(sm), "SMAP L-band soil moisture (AM)", "YlGnBu", 0, 0.5)]):
        im = a.imshow(img, origin="lower", extent=ext, aspect="auto", cmap=cm,
                      vmin=lo, vmax=hi)
        a.set_title(ttl, fontsize=tsz, pad=4)
        a.tick_params(labelsize=tick)
        cb = fig.colorbar(im, ax=a, shrink=0.85); cb.ax.tick_params(labelsize=tick)
        a.text(0.015, 0.96, letter, transform=a.transAxes, fontsize=plab,
               fontweight="bold", va="top", ha="left",
               bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                         alpha=0.75, edgecolor="none"))
    fig.suptitle("Surface Wetness Index vs SMAP L-band soil moisture "
                 "(co-located land)", fontsize=tsz * 1.1)
    fig.tight_layout(); fig.savefig(out, dpi=150); print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
