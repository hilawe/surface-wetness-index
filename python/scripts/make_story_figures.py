"""Regenerable statistical figures for the reconstruction paper.

    python -m scripts.make_story_figures [--results DIR] [--out DIR]

Reads only committed result files under results/ and writes four figures:

  fig_calibration.png  firing fraction under uniform 85 GHz offsets, and the
                       per-cell error-model flip and invalidation fractions
  fig_transfer.png     the two preregistered SWAMPS months with block-bootstrap
                       intervals, and the cross-pair interval on their difference
  fig_optical.png      per-region correlation against the optical JRC layer and
                       against SWAMPS on identical cells
  fig_flood.png        the 1998 Ganges and Brahmaputra monthly departures for
                       the index and the reference, with the July-to-September
                       box-car band

Every plotted number is read from the committed artifact, so each figure is
regenerable and auditable against results/ alone.
"""

import json
import os
import sys

import numpy as np


def opt(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def _load(results, name):
    with open(os.path.join(results, name)) as fh:
        return json.load(fh)


def _panel_label(ax, letter, size):
    ax.text(0.015, 0.96, f"({letter})", transform=ax.transAxes,
            fontsize=size, fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      alpha=0.75, edgecolor="none"))


def fig_calibration(results, out, plt):
    d = _load(results, "calibration_sensitivity_F16_20160715.json")
    base = d["baseline_firing_fraction"]
    offs = [-3.0, -1.0, 0.0, 1.0, 3.0]
    frac = [d["bias_perturbation"][f"{o:+.1f}K"]["firing_fraction"]
            if o else base for o in offs]

    w = 8.5
    sc = w / 6.5
    t, lab, tick = 10 * sc, 10 * sc, 9 * sc
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(w, 4.0),
                                   constrained_layout=True)

    ax1.plot(offs, frac, "o-", color="C0", lw=1.5, ms=6)
    ax1.axhline(base, color="0.6", lw=0.8, ls=":")
    ax1.annotate(f"baseline {base:.3f}", xy=(-3, base),
                 xytext=(-2.9, base + 0.012), fontsize=tick, color="0.35")
    ax1.set_xlabel("uniform offset applied to both 85 GHz channels (K)",
                   fontsize=lab)
    ax1.set_ylabel("fraction of cells with WET > 0", fontsize=lab)
    ax1.set_title("Firing fraction under a systematic offset", fontsize=t, pad=4)
    ax1.set_xticks(offs)
    ax1.tick_params(labelsize=tick)
    _panel_label(ax1, "a", 12 * sc)

    # Both components share one denominator, the baseline-valid cells, and
    # stack to the fraction that fails to reproduce its baseline classification
    # by either route. The flip component is rescaled from its conditional form
    # onto that common denominator so the bar total is the committed
    # changed-or-invalidated fraction.
    models = ("independent\nGaussian", "paired\nempirical")
    keys = ("random_perturbation", "paired_residual_perturbation")
    inval = [100 * d[k]["invalidated_frac_mean"] for k in keys]
    total = [100 * d[k]["changed_or_invalidated_frac_mean"] for k in keys]
    flips = [t - i for t, i in zip(total, inval)]
    x = np.arange(2)
    bw = 0.45
    ax2.bar(x, flips, bw, color="C0", label="classification flipped")
    ax2.bar(x, inval, bw, bottom=flips, color="C3", alpha=0.85,
            label="pushed out of the retrieval")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=tick)
    ax2.set_ylabel("percent of baseline-valid cells", fontsize=lab)
    ax2.set_title("Baseline classification not reproduced", fontsize=t, pad=4)
    ax2.set_ylim(0, 30)
    ax2.legend(fontsize=tick, loc="upper right", frameon=False)
    ax2.tick_params(labelsize=tick)
    for xi, f, i, tt in zip(x, flips, inval, total):
        ax2.text(xi, f / 2, f"{f:.1f}", ha="center", va="center",
                 fontsize=tick, color="white")
        ax2.text(xi, f + i / 2, f"{i:.1f}", ha="center", va="center",
                 fontsize=tick, color="white")
        ax2.text(xi, tt + 0.6, f"{tt:.1f} total", ha="center", fontsize=tick)
    _panel_label(ax2, "b", 12 * sc)

    fig.savefig(os.path.join(out, "fig_calibration.png"), dpi=150)
    plt.close(fig)


def fig_transfer(results, out, plt):
    a = _load(results, "swamps_transfer_F16_201607.json")
    b = _load(results, "swamps_transfer_F13_199807.json")
    d = _load(results, "swamps_transfer_difference_F13_F16.json")

    w = 8.5
    sc = w / 6.5
    t, lab, tick = 10 * sc, 10 * sc, 9 * sc
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(w, 3.0), gridspec_kw={"width_ratios": [1.6, 1.0]},
        constrained_layout=True)

    rows = [("F-16, July 2016 (SSMIS)", a), ("F-13, July 1998 (SSM/I)", b)]
    for i, (label, r) in enumerate(rows):
        ci = r["block_bootstrap_area_weighted_spearman"]
        v = r["spearman_area_weighted"]
        ax1.errorbar(v, i, xerr=[[v - ci["lo"]], [ci["hi"] - v]],
                     fmt="o", color="C0", ms=7, capsize=4, lw=1.5)
        ax1.text(v, i + 0.16, f"{v:+.3f}", ha="center", fontsize=tick)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels([r[0] for r in rows], fontsize=tick)
    ax1.set_ylim(-0.6, 1.6)
    ax1.set_xlim(0.45, 0.72)
    ax1.set_xlabel("area-weighted Spearman correlation vs SWAMPS", fontsize=lab)
    ax1.set_title("The two preregistered months", fontsize=t, pad=4)
    ax1.tick_params(labelsize=tick)
    _panel_label(ax1, "a", 12 * sc)

    pd = d["point_difference_a_minus_b"]
    lo = d["difference_ci_95"]["lo"]
    hi = d["difference_ci_95"]["hi"]
    ax2.errorbar(pd, 0, xerr=[[pd - lo], [hi - pd]], fmt="o", color="C0",
                 ms=7, capsize=4, lw=1.5)
    ax2.axvline(0.0, color="0.4", lw=0.9)
    ax2.text(pd, 0.16, f"{pd:+.3f}", ha="center", fontsize=tick)
    ax2.set_yticks([0])
    ax2.set_yticklabels(["1998 minus 2016"], fontsize=tick)
    ax2.set_ylim(-0.6, 1.6)
    ax2.set_xlim(-0.15, 0.20)
    ax2.set_xlabel("difference", fontsize=lab)
    ax2.set_title("F-13 minus F-16", fontsize=t, pad=4)
    ax2.tick_params(labelsize=tick)
    _panel_label(ax2, "b", 12 * sc)

    fig.savefig(os.path.join(out, "fig_transfer.png"), dpi=150)
    plt.close(fig)


def fig_optical(results, out, plt):
    d = _load(results, "optical_independence_test.json")
    order = [("90E_30N", "Ganges and Brahmaputra"),
             ("60E_70N", "Western Siberia, boreal"),
             ("100W_50N", "North American prairie"),
             ("70W_0N", "Amazon"),
             ("0E_30N", "Sahara, arid control")]

    w = 7.5
    sc = w / 6.5
    t, lab, tick = 10 * sc, 10 * sc, 9 * sc
    fig, ax = plt.subplots(figsize=(w, 3.8))
    ys = np.arange(len(order))[::-1]
    for y, (key, label) in zip(ys, order):
        tile = d["tiles"][key]
        vo = tile["vs_optical"]["spearman_area_weighted"]
        vs = tile["vs_swamps"]["spearman_area_weighted"]
        ax.plot([vo, vs], [y, y], color="0.75", lw=1.2, zorder=1)
        ax.plot(vo, y, "o", color="C1", ms=8, zorder=2)
        ax.plot(vs, y, "s", color="C0", ms=8, zorder=2)
    ax.axvline(0.0, color="0.4", lw=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels([label for _, label in order], fontsize=tick)
    ax.set_xlim(-0.08, 1.0)
    ax.set_xlabel("area-weighted Spearman correlation, identical cells",
                  fontsize=lab)
    ax.set_title("The index against an optical and a microwave reference, "
                 "July 2016", fontsize=t, pad=5)
    ax.plot([], [], "o", color="C1", label="vs optical JRC occurrence")
    ax.plot([], [], "s", color="C0", label="vs SWAMPS fractional water")
    ax.legend(fontsize=tick, loc="lower right", frameon=False)
    ax.tick_params(labelsize=tick)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_optical.png"), dpi=150)
    plt.close(fig)


def fig_flood(results, out, plt):
    d = _load(results, "flood_event_test_1998.json")
    g = d["regions"]["ganges_brahmaputra_1998"]
    months = d["months"]
    x = np.arange(12)

    w = 7.5
    sc = w / 6.5
    t, lab, tick = 10 * sc, 10 * sc, 9 * sc
    fig, ax = plt.subplots(figsize=(w, 3.8))
    ax.axvspan(5.5, 8.5, color="0.88", zorder=0)
    ax.text(7.0, 5.15, "July to September box-car", ha="center",
            fontsize=tick, color="0.35")
    l1, = ax.plot(x, g["index_anomaly_series"], "o-", color="C0", lw=1.5,
                  label="index departure (left axis)")
    ax.axhline(0.0, color="0.4", lw=0.9)
    ax.set_ylabel("index departure from 1998 mean", fontsize=lab)
    ax.set_ylim(-4.8, 5.6)
    ax2 = ax.twinx()
    l2, = ax2.plot(x, g["reference_anomaly_series"], "s--", color="C2", lw=1.5,
                   label="SWAMPS departure (right axis)")
    ax2.set_ylabel("fractional water departure from 1998 mean", fontsize=lab)
    ax2.set_ylim(-0.16, 0.187)
    ax.set_xticks(x)
    ax.set_xticklabels(months, fontsize=tick)
    ax.tick_params(labelsize=tick)
    ax2.tick_params(labelsize=tick)
    ax.set_title("Ganges and Brahmaputra, 1998 monthly departures", fontsize=t,
                 pad=5)
    r_ir = d["regions"]["ganges_brahmaputra_1998"]["regional_series_correlation"]
    r_ib = g["index_vs_seasonal_boxcar"]
    r_rb = g["reference_vs_seasonal_boxcar"]
    ax.text(0.985, 0.03,
            f"r(index, reference) = {r_ir:+.2f}\n"
            f"r(index, box-car) = {r_ib:+.2f}\n"
            f"r(reference, box-car) = {r_rb:+.2f}",
            transform=ax.transAxes, fontsize=tick, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      alpha=0.85, edgecolor="0.7"))
    ax.legend(handles=[l1, l2], fontsize=tick, loc="upper left",
              bbox_to_anchor=(0.02, 0.97), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_flood.png"), dpi=150)
    plt.close(fig)


def main():
    argv = sys.argv
    results = opt(argv, "--results", "../results")
    out = opt(argv, "--out", "../docs/figures")
    os.makedirs(out, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    fig_calibration(results, out, plt)
    fig_transfer(results, out, plt)
    fig_optical(results, out, plt)
    fig_flood(results, out, plt)
    for name in ("fig_calibration", "fig_transfer", "fig_optical", "fig_flood"):
        print(f"wrote {os.path.join(out, name + '.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
