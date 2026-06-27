"""Derive and check the 85-to-91 GHz calibration from overlap-era files.

Pairs are alternating SSM/I (true 85) and SSMIS (91) files for the same date(s):

    python -m scripts.calibrate_8591 --pass dsc \
        CSU_SSMI_..._F13_D20060601.nc  CSU_SSMIS_..._F16_D20060601.nc \
        CSU_SSMI_..._F13_D20060901.nc  CSU_SSMIS_..._F16_D20060901.nc

Options:
    --linear        per-channel linear fit (default is multi-channel)
    --apply FILE    apply the fit to an SSMIS file and compare the wetness
                    product with vs without the adjustment
    --png FILE      write a 85-vs-91 scatter and residual plot
"""

import argparse
import sys

import numpy as np

from swi import calib_8591 as cal


def report(f):
    print(f"\nfit: model={f['model']}  n={f['n']:,} land cells")
    print(f"  naive bias (mean 91 - 85):  V {f['raw_bias_v']:+.2f} K   "
          f"H {f['raw_bias_h']:+.2f} K")
    print(f"  85V model: coef={np.array2string(f['coef_v'], precision=4)}  "
          f"rms={f['stats_v']['rms']:.2f} K  bias={f['stats_v']['bias']:+.2f}  "
          f"r2={f['stats_v']['r2']:.3f}  (in-sample)")
    print(f"  85H model: coef={np.array2string(f['coef_h'], precision=4)}  "
          f"rms={f['stats_h']['rms']:.2f} K  bias={f['stats_h']['bias']:+.2f}  "
          f"r2={f['stats_h']['r2']:.3f}  (in-sample)")
    if "cv_stats_v" in f:
        print(f"  85V CV ({f.get('cv_k', '?')}-fold):  "
              f"rms={f['cv_stats_v']['rms']:.2f} K  "
              f"bias={f['cv_stats_v']['bias']:+.2f}  "
              f"r2={f['cv_stats_v']['r2']:.3f}  (held-out, honest)")
        print(f"  85H CV ({f.get('cv_k', '?')}-fold):  "
              f"rms={f['cv_stats_h']['rms']:.2f} K  "
              f"bias={f['cv_stats_h']['bias']:+.2f}  "
              f"r2={f['cv_stats_h']['r2']:.3f}  (held-out, honest)")


def compare_product(ssmis_file, f, pass_):
    from swi import io_csu_grid as io, core_numpy
    lat, lon, tb, sensor = io.read_channels(ssmis_file, pass_=pass_)
    valid = np.isfinite(tb).all(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = core_numpy.evaluate_kelvin(tb[valid])
        adj = core_numpy.evaluate_kelvin(cal.apply(tb[valid], f))
    def line(tag, r):
        return (f"  {tag:10s} WET>0={int((r.wet>0).sum()):>7,}  "
                f"SNOW>0={int((r.snow>0).sum()):>7,}  "
                f"RTEMP={int((r.temp>-90).sum()):>7,}")
    print(f"\napply to {ssmis_file.split('/')[-1]} ({pass_}):")
    print(line("raw 91", raw))
    print(line("85-adj", adj))
    dw = (adj.wet > 0).sum() - (raw.wet > 0).sum()
    print(f"  change in wet-surface detections: {dw:+,} "
          f"({100*dw/max(1,(raw.wet>0).sum()):+.1f}%)")


def make_plot(c, f, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    s = slice(None, None, max(1, c["n"] // 40000))     # thin for plotting
    ax[0].scatter(c["t91v"][s], c["t85v"][s], s=2, alpha=0.2)
    lim = [min(c["t91v"].min(), c["t85v"].min()), max(c["t91v"].max(), c["t85v"].max())]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set_xlabel("SSMIS 91V (K)"); ax[0].set_ylabel("SSM/I 85V (K)")
    ax[0].set_title(f"85V vs 91V over land (n={c['n']:,})")
    pred = (f["coef_v"][0] * c["t91v"] + f["coef_v"][1] * c["v22"]
            + f["coef_v"][2] * c["v37"] + f["coef_v"][3]) if f["model"] == "multi" \
        else f["coef_v"][0] * c["t91v"] + f["coef_v"][1]
    ax[1].hist((c["t85v"] - pred), bins=80)
    ax[1].set_xlabel("85V residual: true - predicted (K)")
    ax[1].set_title(f"residual  rms={f['stats_v']['rms']:.2f} K")
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="alternating SSM/I and SSMIS files")
    ap.add_argument("--pass", dest="pass_", default="dsc", choices=["asc", "dsc"])
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--apply", dest="apply_file")
    ap.add_argument("--png")
    ap.add_argument("--save", help="write the fit as JSON to this path")
    ap.add_argument("--cv-k", type=int, default=5,
                    help="k-fold cross-validation folds for the honest fit "
                    "quality estimate. The saved fit uses all data; CV is the "
                    "diagnostic. Set 0 to skip CV and only report in-sample.")
    a = ap.parse_args()
    if len(a.files) % 2 != 0:
        ap.error("provide pairs: SSM/I then SSMIS, even count")
    pairs = list(zip(a.files[0::2], a.files[1::2]))
    print(f"pooling {len(pairs)} pair(s), pass={a.pass_}")
    c = cal.pool(pairs, pass_=a.pass_)
    if a.cv_k and a.cv_k >= 2:
        f = cal.cross_validated_fit(c, multi=not a.linear, k=a.cv_k)
    else:
        f = cal.fit(c, multi=not a.linear)
    report(f)
    if a.save:
        from swi import calib_8591 as _cal
        _cal.save_fit(f, a.save, meta={"pairs": pairs, "pass": a.pass_,
                                       "model": f["model"]})
        print(f"\nsaved fit to {a.save}")
    if a.apply_file:
        with np.errstate(divide="ignore", invalid="ignore"):
            compare_product(a.apply_file, f, a.pass_)
    if a.png:
        make_plot(c, f, a.png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
