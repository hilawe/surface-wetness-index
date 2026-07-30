"""Bootstrap interval for the DIFFERENCE between two SWAMPS transfer correlations.

    python -m scripts.swamps_transfer_difference A.json B.json [--out J.json]

Overlap between two separately bootstrapped intervals is not a test of the
difference between the statistics, so this driver computes one directly. Each
input is a swamps_transfer_test result whose block bootstrap saved its draws.
The two months are independent samples, so a draw of the difference pairs one
replicate from each file, and the percentile interval of those differences is
the interval for the difference of the two area-weighted Spearman correlations.

This statistic was added after the preregistered comparison was run and is
labelled post hoc in its output.
"""

import json
import os
import sys

import numpy as np


def opt(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def main():
    argv = sys.argv
    if len(argv) < 3:
        print(__doc__)
        return 1
    path_a, path_b = argv[1], argv[2]
    out_path = opt(argv, "--out", "../results/swamps_transfer_difference.json")

    with open(path_a) as fh:
        a = json.load(fh)
    with open(path_b) as fh:
        b = json.load(fh)
    da = np.asarray(a["block_bootstrap_area_weighted_spearman"]["draws"], np.float64)
    db = np.asarray(b["block_bootstrap_area_weighted_spearman"]["draws"], np.float64)
    if da.size == 0 or db.size == 0:
        print("one of the inputs carries no saved bootstrap draws; rerun "
              "swamps_transfer_test with the draw-saving version first")
        return 1
    n = min(da.size, db.size)
    d = da[:n] - db[:n]
    good = d[np.isfinite(d)]
    lo, hi = np.percentile(good, [2.5, 97.5])

    res = {
        "note": "post hoc; not part of the preregistered comparison",
        "a": {"label": a["label"],
              "spearman_area_weighted": a["spearman_area_weighted"]},
        "b": {"label": b["label"],
              "spearman_area_weighted": b["spearman_area_weighted"]},
        "point_difference_a_minus_b": (a["spearman_area_weighted"]
                                       - b["spearman_area_weighted"]),
        "difference_ci_95": {"lo": float(lo), "hi": float(hi),
                             "n_pairs": int(good.size)},
        "frac_gt_0": float((good > 0).mean()),
        "spans_zero": bool(lo < 0.0 < hi),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print(f"{a['label']} minus {b['label']}: "
          f"{res['point_difference_a_minus_b']:+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}] 95 percent, spans zero: {res['spans_zero']}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
