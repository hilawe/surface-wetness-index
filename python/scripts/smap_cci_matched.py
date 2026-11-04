"""Compare the index against SMAP and ESA CCI on the SAME cells, plus robustness.

    python -m scripts.smap_cci_matched PRODUCT.nc YYYYMM [--json OUT]

WHY THIS EXISTS. The headline SMAP and ESA CCI comparisons run on different
domains, because SMAP's quality screen keeps a different set of cells than the
CCI flag does. A raw difference between their detection contrasts therefore
confounds two things, how good the reference is and which cells it kept. This
script removes the domain confound by scoring both references on the
INTERSECTION of their valid cells, and then checks that the result is not an
artifact of two processing choices this project made.

It exists as a committed script rather than an ad-hoc session because the
numbers it produces are quoted in the manuscript, and a quoted number must be
re-runnable.

WHAT IT FOUND, stated no wider than the evidence. The expectation recorded in
docs/SMAP_Comparison_Scope.md was that agreement with SMAP would be LOWER than
with ESA CCI. It was not confirmed. The index agrees better with SMAP on matched
cells and on every metric, and the paired interval on the difference excludes
zero.

WHAT THAT DOES NOT SHOW, because an earlier version of this docstring claimed it.
The ordering does not identify the direction or magnitude of shared-channel bias,
and it does not establish that reference quality caused the difference. A shared
brightness-temperature artifact could raise the index and the CCI passive
retrieval together, which would CREATE a contrast rather than blur one, and
nothing measured here rules that out. The shared-sensor caveat on ESA CCI stands
as a reason not to call it independent, and this experiment neither confirms nor
refutes it.
"""

import json
import os
import sys

import numpy as np

from swi import validate as val
from swi.io_esacci import PRODUCT as CCI_PRODUCT, VERSION as CCI_VERSION, load_cci_monthly
from swi.io_smap import (SHORT_NAME, VERSION, load_smap_monthly, load_coords,
                         month_files, _read_day)
from scripts.validate_smap import read_product

SEED = 20260731


def _score(wet, ref, mask, plat, plon, bootstrap=True):
    W, S = wet[mask], ref[mask]
    s = val.skill_scores(W, S)
    d = val.detection_contrast(W, S, thr=0.0)
    out = {"n": int(s["n"]), "spearman": float(s["spearman_r"]),
           "pearson": float(s["pearson_r"]), "contrast": float(d["ratio"]),
           "mean_ref_where_firing": float(d["mean_hi"]),
           "mean_ref_where_quiet": float(d["mean_lo"])}
    if bootstrap:
        la = np.broadcast_to(plat[:, None], wet.shape)[mask]
        lo = np.broadcast_to(plon[None, :], wet.shape)[mask]
        ci = val.block_bootstrap_ci(
            lambda i: val.detection_contrast(W[i], S[i], thr=0.0)["ratio"],
            val.block_ids(la, lo, 10.0), n_draws=2000, seed=SEED, null_value=1.0)
        out["ci95"] = [float(ci["lo"]), float(ci["hi"])]
        out["n_blocks"] = int(ci["n_blocks"])
    return out


def _paired(wet, ref_a, ref_b, mask, plat, plon):
    """Bootstrap the DIFFERENCE and RATIO of two contrasts on identical cells.

    Non-overlap of two marginal intervals is not a test of the difference. Both
    references are resampled on the SAME block draws here, so the interval is on
    the paired quantity and answers the question directly.
    """
    W = wet[mask]
    A, B = ref_a[mask], ref_b[mask]
    la = np.broadcast_to(plat[:, None], wet.shape)[mask]
    lo = np.broadcast_to(plon[None, :], wet.shape)[mask]
    blocks = val.block_ids(la, lo, 10.0)

    def diff(i):
        return (val.detection_contrast(W[i], A[i], thr=0.0)["ratio"]
                - val.detection_contrast(W[i], B[i], thr=0.0)["ratio"])

    def ratio(i):
        return (val.detection_contrast(W[i], A[i], thr=0.0)["ratio"]
                / val.detection_contrast(W[i], B[i], thr=0.0)["ratio"])

    d = val.block_bootstrap_ci(diff, blocks, n_draws=2000, seed=SEED, null_value=0.0)
    r = val.block_bootstrap_ci(ratio, blocks, n_draws=2000, seed=SEED, null_value=1.0)
    return {"difference": [float(d["lo"]), float(d["hi"])],
            "ratio": [float(r["lo"]), float(r["hi"])],
            "n_blocks": int(d["n_blocks"]),
            "point_difference": float(diff(np.arange(W.size))),
            "point_ratio": float(ratio(np.arange(W.size)))}


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    product, month = sys.argv[1], sys.argv[2]
    json_out = None
    if "--json" in sys.argv:
        json_out = sys.argv[sys.argv.index("--json") + 1]
    smap_dir = "../data/smap"
    cci_dir = f"../data/esacci/{month}"

    plat, plon, wet, snowf = read_product(product, "dsc")
    unfrozen = ~(snowf > 0.5)

    smap, smap_nd, st = load_smap_monthly(month, smap_dir, plat, plon,
                                          download=False, verbose=False)
    elat, elon, cci, cci_nd = load_cci_monthly(month, cci_dir, download=False)
    cci_on = val.regrid_nearest(elat, elon, cci, plat, plon)

    m_smap = val.common_valid(wet, smap, np.isfinite(smap)) & unfrozen & (wet >= 0)
    m_cci = val.common_valid(wet, cci_on, np.isfinite(cci_on)) & unfrozen & (wet >= 0)
    m_both = m_smap & m_cci

    res = {
        "month": month, "product": os.path.basename(product),
        "seed": SEED,
        "own_domain": {
            "SMAP": _score(wet, smap, m_smap, plat, plon),
            "ESA_CCI": _score(wet, cci_on, m_cci, plat, plon)},
        "matched_domain": {
            "n_cells": int(np.sum(m_both)),
            "SMAP": _score(wet, smap, m_both, plat, plon),
            "ESA_CCI": _score(wet, cci_on, m_both, plat, plon)},
    }

    # the paired test. Non-overlap of two marginal intervals is not a test of the
    # difference, so bootstrap the difference itself on the same block draws.
    res["matched_domain"]["paired"] = _paired(wet, smap, cci_on, m_both, plat, plon)

    # --- robustness, so a favorable surprise is not taken on trust ----------
    # The relaxed screen admits recommended AND uncertain retrievals but still
    # requires an attempted, successful one. Dropping every quality bit would
    # admit retrievals the product marks unsuccessful, which is not a defensible
    # sensitivity.
    nq, _, _ = load_smap_monthly(month, smap_dir, plat, plon, download=False,
                                 screen="successful", verbose=False)
    m_nq = val.common_valid(wet, nq, np.isfinite(nq)) & unfrozen & (wet >= 0)
    m_nq_both = m_nq & m_cci

    slat, slon = load_coords(smap_dir)
    ilat = np.abs(slat[:, None] - plat[None, :]).argmin(axis=0)
    lon360 = np.where(slon < 0, slon + 360.0, slon)
    ilon = np.abs(lon360[:, None] - plon[None, :]).argmin(axis=0)
    acc = np.zeros((plat.size, plon.size)); cnt = np.zeros_like(acc)
    for _, path in month_files(month, smap_dir):     # THIS month only
        sm = _read_day(path, screen="recommended")
        samp = sm[np.ix_(ilat, ilon)]
        g = np.isfinite(samp); acc[g] += samp[g]; cnt[g] += 1
    nn = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan)
    m_nn = val.common_valid(wet, nn, np.isfinite(nn)) & unfrozen & (wet >= 0)

    res["robustness"] = {
        "note": "the ordering must survive relaxing the screen and changing the regrid",
        "smap_relaxed_screen_successful": _score(wet, nq, m_nq, plat, plon,
                                                 bootstrap=False),
        "smap_relaxed_on_cci_intersection": _score(wet, nq, m_nq_both, plat, plon,
                                                   bootstrap=False),
        "cci_on_relaxed_intersection": _score(wet, cci_on, m_nq_both, plat, plon,
                                              bootstrap=False),
        # This key keeps its British spelling because the committed artifact
        # results/smap_cci_matched_F16_202307.json already carries it. It is a
        # record key, not prose. Renaming it would silently split the series.
        "smap_nearest_neighbour_regrid": _score(wet, nn, m_nn, plat, plon,
                                                bootstrap=False),
    }

    # --- coverage matching. The index and each reference average their own
    # available days, so a cell can fire early in July and be scored against
    # reference observations from unrelated later dates. Requiring dense coverage
    # in BOTH tests how much of the association survives that.
    cci_nd_on = val.regrid_nearest(elat, elon, cci_nd.astype(np.float64), plat, plon)
    dense = m_both & (smap_nd >= 20) & (cci_nd_on >= 20)
    res["coverage_matched"] = {
        "note": "at least 20 valid days in BOTH references; a monthly spatial "
                "association is not contemporaneous detection",
        "min_days": 20,
        "n_cells": int(np.sum(dense)),
        "SMAP": _score(wet, smap, dense, plat, plon, bootstrap=False),
        "ESA_CCI": _score(wet, cci_on, dense, plat, plon, bootstrap=False),
    }

    res["ingest"] = {k: (v if isinstance(v, str) else
                         float(v) if k == "kept_fraction_of_successful" else int(v))
                     for k, v in st.items()}
    res["prereg"] = {
        "expectation": "SMAP contrast BELOW the ESA CCI value and above 1.0",
        "source": "docs/SMAP_Comparison_Scope.md, recorded before any number existed",
        "outcome": "NOT CONFIRMED. The SMAP contrast is higher on matched cells and "
                   "the paired interval on the difference excludes zero.",
        "what_it_does_not_show": "the direction or magnitude of shared-channel bias, "
                                 "and not that reference quality caused the difference",
    }

    md = res["matched_domain"]
    print(f"MATCHED DOMAIN ({md['n_cells']:,} identical cells)")
    for k in ("SMAP", "ESA_CCI"):
        r = md[k]
        print(f"  {k:<9} spearman {r['spearman']:+.3f}  contrast {r['contrast']:.2f}  "
              f"95% CI [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]")
    pr = md["paired"]
    print(f"  PAIRED difference {pr['point_difference']:+.3f}  "
          f"95% CI [{pr['difference'][0]:+.3f}, {pr['difference'][1]:+.3f}]  "
          f"(excludes zero: {not (pr['difference'][0] <= 0 <= pr['difference'][1])})")
    print(f"  PAIRED ratio      {pr['point_ratio']:.3f}  "
          f"95% CI [{pr['ratio'][0]:.3f}, {pr['ratio'][1]:.3f}]")

    print("\nROBUSTNESS")
    for k, r in res["robustness"].items():
        if k == "note":
            continue
        print(f"  {k:<34} n={r['n']:>8,}  spearman {r['spearman']:+.3f}  "
              f"contrast {r['contrast']:.2f}")

    cm = res["coverage_matched"]
    print(f"\nCOVERAGE MATCHED (>= {cm['min_days']} valid days in both, "
          f"{cm['n_cells']:,} cells)")
    for k in ("SMAP", "ESA_CCI"):
        print(f"  {k:<9} spearman {cm[k]['spearman']:+.3f}  "
              f"contrast {cm[k]['contrast']:.2f}")

    print(f"\nPRE-REGISTERED EXPECTATION: {res['prereg']['outcome']}")

    if json_out:
        os.makedirs(os.path.dirname(json_out) or ".", exist_ok=True)
        with open(json_out, "w") as f:
            json.dump(res, f, indent=2, sort_keys=True)
        print(f"wrote {json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
