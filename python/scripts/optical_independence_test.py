"""An optically independent check of the surface-water claim.

Every inundation result elsewhere in this project uses SWAMPS, which is built from
combined active and passive microwave observations, while the index is a
passive-microwave retrieval. Two references drawn from the same measurement family can
agree partly because they share an error structure, so that agreement is not
independent confirmation. This driver runs the test fixed in
`docs/Preregistration_Optical_Independence_Test.md` using a reference from a different
observation family entirely: the Joint Research Centre Global Surface Water occurrence
layer, derived from the Landsat optical archive at 30 m.

The occurrence layer reports, per 30 m pixel, the percentage of valid observations
between 1984 and 2021 in which surface water was detected. Averaging it over a product
grid cell gives a long-term water fraction, which is the right reference for the claim
under test: where the index finds water, not when.

The decisive number is not the correlation against the optical reference on its own. It
is that correlation set beside the SWAMPS correlation computed on exactly the same
cells, because that comparison is what separates real skill from shared-family
agreement.

Usage:
    python -m scripts.optical_independence_test PRODUCT.nc SWAMPS_FW.nc
        [--jrc-dir DIR] [--out J.json]
"""

import glob
import json
import os
import sys

import numpy as np

from swi import grids, validate as val

# Fixed in the preregistration. Each tile is the 10-degree square whose north-west
# corner is named, which is the JRC tiling convention.
TILES = {
    "90E_30N": "Ganges and Brahmaputra, Southeast Asia (humid tropics)",
    "60E_70N": "Western Siberia (boreal)",
    "100W_50N": "North American prairie pothole (temperate)",
    "0E_30N": "Sahara (arid)",
    "70W_0N": "Amazon (humid tropics, dense canopy)",
}
WET_THR = 0.0          # index fires
FW_INUNDATED = 0.05    # reference counted as inundated, same as elsewhere
BLOCK_DEG = 10.0


def opt(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def _tile_bounds(name):
    """(lat_south, lat_north, lon_west, lon_east) in 0..360 for a JRC tile name."""
    lon_s, lat_s = name.split("_")
    lon = float(lon_s[:-1]) * (1 if lon_s[-1] == "E" else -1)
    lat = float(lat_s[:-1]) * (1 if lat_s[-1] == "N" else -1)
    # named corner is north-west; tiles are 10 degrees on a side
    return lat - 10.0, lat, lon % 360.0, (lon + 10.0) % 360.0


def _aggregate_tile(path, dst_lat, dst_lon):
    """Mean JRC occurrence (as a 0..1 fraction) on the destination grid.

    The two grids align exactly: the destination cell size (0.25 degree) is an
    integer multiple of the source pixel (0.00025 degree), 1000 source pixels to a
    side, and the 10-degree tile edges fall on destination cell edges. So the
    aggregation is an exact block reduction rather than a scatter, which matters
    because a tile holds 1.6 billion pixels. Read one destination row of source
    pixels at a time so the tile is never held in memory in full.
    """
    import rasterio

    dlat = float(dst_lat[1] - dst_lat[0])
    dlon = float(dst_lon[1] - dst_lon[0])
    sum_ = np.zeros((dst_lat.size, dst_lon.size), np.float64)
    cnt = np.zeros((dst_lat.size, dst_lon.size), np.int64)

    with rasterio.open(path) as src:
        tr = src.transform
        px = abs(float(tr.a))
        block = int(round(dlat / px))                  # source rows per dest row
        if abs(block * px - dlat) > 1e-9 or src.width % block:
            raise ValueError(f"{path}: source and destination grids do not align")
        ncol = src.width // block
        north = float(tr.f)                            # tile north edge
        west = float(tr.c)                             # tile west edge
        for r0 in range(0, src.height, block):
            if r0 + block > src.height:
                break
            a = src.read(1, window=((r0, r0 + block), (0, src.width)))
            valid = a != 255                           # 255 = no valid observation
            # occurrence is a percentage; convert to a fraction
            v = np.where(valid, a, 0).astype(np.float64) / 100.0
            s_sum = v.reshape(block, ncol, block).sum(axis=(0, 2))
            s_cnt = valid.reshape(block, ncol, block).sum(axis=(0, 2))
            # this source row block spans one destination row, south edge first
            south_edge = north - (r0 + block) * px
            iy = int(round((south_edge - (dst_lat[0] - dlat / 2.0)) / dlat))
            if not (0 <= iy < dst_lat.size):
                continue
            for k in range(ncol):
                west_edge = (west + k * block * px) % 360.0
                ix = int(round((west_edge - (dst_lon[0] - dlon / 2.0)) / dlon))
                if 0 <= ix < dst_lon.size and s_cnt[k] > 0:
                    sum_[iy, ix] += s_sum[k]
                    cnt[iy, ix] += s_cnt[k]
    out = np.full(sum_.shape, np.nan)
    good = cnt > 0
    out[good] = sum_[good] / cnt[good]
    return out, cnt


def _stats(index, ref, w, lat, lon, label):
    s = val.skill_scores(index, ref)
    sp_w = val.weighted_spearman(index, ref, w)
    dc = val.detection_contrast(index, ref, thr=WET_THR)
    dc_w = val.weighted_detection_contrast(index, ref, w, thr=WET_THR)
    cat = val.categorical(index, ref, a_hi=WET_THR, b_hi=FW_INUNDATED)
    return {
        "label": label, "n": int(s["n"]),
        "spearman": s["spearman_r"], "spearman_area_weighted": sp_w,
        "pearson": s["pearson_r"],
        "contrast": dc["ratio"], "contrast_area_weighted": dc_w["ratio"],
        "contingency": cat,
    }


def main():
    argv = sys.argv
    if len(argv) < 3:
        print(__doc__)
        return 1
    product, swamps = argv[1], argv[2]
    jrc_dir = opt(argv, "--jrc-dir", "../data/jrc_gsw")
    out_path = opt(argv, "--out", "../results/optical_independence_test.json")

    from scripts.validate_swamps import load_swamps_fw, read_product

    plat, plon, wet, snowf = read_product(product)
    slat, slon, fw = load_swamps_fw(swamps)
    swamps_on = val.regrid_nearest(slat, slon, fw, plat, plon)
    land = grids.land_mask(plat, plon)
    if land.all():
        raise RuntimeError("land mask is all-True; install global_land_mask.")

    lat2d = np.broadcast_to(plat[:, None], wet.shape)
    lon2d = np.broadcast_to(plon[None, :], wet.shape)

    out = {"product": os.path.basename(product), "reference_swamps": os.path.basename(swamps),
           "frozen_thresholds": {"wet": WET_THR, "inundated": FW_INUNDATED},
           "tiles": {}}
    pooled = {"i": [], "j": [], "s": [], "w": [], "lat": [], "lon": []}

    for tile, note in TILES.items():
        path = os.path.join(jrc_dir, f"occurrence_{tile}.tif")
        if not os.path.exists(path):
            print(f"  {tile}: MISSING {path}")
            continue
        s_lat, n_lat, w_lon, e_lon = _tile_bounds(tile)
        jrc, cnt = _aggregate_tile(path, plat, plon)
        box = (lat2d >= s_lat) & (lat2d <= n_lat)
        box &= ((lon2d >= w_lon) & (lon2d <= e_lon)) if w_lon < e_lon else \
               ((lon2d >= w_lon) | (lon2d <= e_lon))
        m = (box & land & np.isfinite(jrc) & np.isfinite(swamps_on)
             & (wet >= 0) & ~(snowf > 0.5) & (cnt > 0))
        n = int(m.sum())
        if n < 100:
            print(f"  {tile}: only {n} usable cells, skipped")
            continue
        I, J, S = wet[m], jrc[m], swamps_on[m]
        LA, LO = lat2d[m], lon2d[m]
        W = np.cos(np.deg2rad(LA))
        out["tiles"][tile] = {
            "note": note, "n_cells": n,
            "vs_optical": _stats(I, J, W, LA, LO, "JRC optical"),
            "vs_swamps": _stats(I, S, W, LA, LO, "SWAMPS microwave"),
            "optical_vs_swamps_spearman":
                val.skill_scores(J, S)["spearman_r"],
        }
        for k, v in zip(pooled, (I, J, S, W, LA, LO)):
            pooled[k].append(v)
        o = out["tiles"][tile]
        print(f"\n  {tile}  {note}")
        print(f"    cells {n:,}")
        print(f"    index vs optical  : Spearman {o['vs_optical']['spearman']:+.3f} "
              f"(area-wt {o['vs_optical']['spearman_area_weighted']:+.3f})  "
              f"contrast {o['vs_optical']['contrast']:.2f}x")
        print(f"    index vs SWAMPS   : Spearman {o['vs_swamps']['spearman']:+.3f} "
              f"(area-wt {o['vs_swamps']['spearman_area_weighted']:+.3f})  "
              f"contrast {o['vs_swamps']['contrast']:.2f}x")
        print(f"    optical vs SWAMPS : Spearman {o['optical_vs_swamps_spearman']:+.3f}")

    if pooled["i"]:
        I = np.concatenate(pooled["i"]); J = np.concatenate(pooled["j"])
        S = np.concatenate(pooled["s"]); W = np.concatenate(pooled["w"])
        LA = np.concatenate(pooled["lat"]); LO = np.concatenate(pooled["lon"])
        po = _stats(I, J, W, LA, LO, "JRC optical")
        ps = _stats(I, S, W, LA, LO, "SWAMPS microwave")
        blocks = val.block_ids(LA, LO, BLOCK_DEG)
        ci_o = val.block_bootstrap_ci(
            lambda idx: val.weighted_spearman(I[idx], J[idx], W[idx]),
            blocks, n_draws=500, seed=20260729)
        out["pooled"] = {"vs_optical": po, "vs_swamps": ps,
                         "optical_vs_swamps_spearman": val.skill_scores(J, S)["spearman_r"],
                         "optical_block_bootstrap_ci": ci_o,
                         "ratio_optical_to_swamps_area_weighted":
                             float(po["spearman_area_weighted"] /
                                   ps["spearman_area_weighted"])
                             if ps["spearman_area_weighted"] else float("nan")}
        # Within-tile values are the headline. Pooling regions whose water content
        # differs by orders of magnitude manufactures between-region variance, and
        # the pooled correlation can exceed every individual tile, which is a
        # Simpson's-paradox artifact rather than skill. Both are reported, with the
        # per-tile medians first.
        t_o = [v["vs_optical"]["spearman_area_weighted"] for v in out["tiles"].values()]
        t_s = [v["vs_swamps"]["spearman_area_weighted"] for v in out["tiles"].values()]
        out["within_tile_median"] = {
            "vs_optical": float(np.median(t_o)), "vs_swamps": float(np.median(t_s)),
            "ratio": float(np.median(t_o) / np.median(t_s)) if np.median(t_s) else float("nan"),
        }
        print(f"\n=== within-tile medians over {len(out['tiles'])} tiles (the headline) ===")
        print(f"  index vs optical  : area-weighted Spearman "
              f"{out['within_tile_median']['vs_optical']:+.3f}")
        print(f"  index vs SWAMPS   : area-weighted Spearman "
              f"{out['within_tile_median']['vs_swamps']:+.3f}")
        print(f"  ratio optical/SWAMPS : {out['within_tile_median']['ratio']:.2f}")
        if po["spearman_area_weighted"] > max(t_o):
            print("  NOTE: the pooled value below exceeds every individual tile, which is\n"
                  "        a between-region pooling effect and not skill. Use the medians.")
        print(f"\n=== pooled over {len(out['tiles'])} tiles, {po['n']:,} cells "
              f"(reported for completeness, see note) ===")
        print(f"  index vs optical  : Spearman {po['spearman']:+.3f} "
              f"(area-wt {po['spearman_area_weighted']:+.3f}, 95% CI "
              f"[{ci_o['lo']:+.3f}, {ci_o['hi']:+.3f}])  contrast "
              f"{po['contrast']:.2f}x (area-wt {po['contrast_area_weighted']:.2f}x)")
        print(f"  index vs SWAMPS   : Spearman {ps['spearman']:+.3f} "
              f"(area-wt {ps['spearman_area_weighted']:+.3f})  contrast "
              f"{ps['contrast']:.2f}x (area-wt {ps['contrast_area_weighted']:.2f}x)")
        print(f"  optical vs SWAMPS : Spearman "
              f"{out['pooled']['optical_vs_swamps_spearman']:+.3f}")
        print(f"  ratio optical/SWAMPS (area-weighted Spearman): "
              f"{out['pooled']['ratio_optical_to_swamps_area_weighted']:.2f}")
        print(f"  contingency vs optical: POD={po['contingency']['POD']:.2f} "
              f"FAR={po['contingency']['FAR']:.2f} CSI={po['contingency']['CSI']:.2f} "
              f"HSS={po['contingency']['HSS']:.2f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
