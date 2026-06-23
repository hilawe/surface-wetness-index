"""Cell-by-cell regression check: NumPy port vs the C oracle.

Generates a large sample of channel values (a mix of physically-structured
cells, which exercise the deep branches, and raw-uniform cells, which exercise
the bad-data filters), runs both engines, and reports any disagreement.

Usage:
    python -m scripts.regression_check [N] [SEED]
"""

import sys

import numpy as np

from swi import core_c, core_numpy
from swi.channels import CHANNEL_NAMES, kelvin_to_packed


def make_sample(n, seed):
    rng = np.random.default_rng(seed)
    n_struct = n // 3
    n_wet = n // 3
    n_unif = n - n_struct - n_wet

    def noise(m, lo, hi):
        return rng.uniform(lo, hi, size=m)

    # Structured: a base temperature with small, mostly physical polarization
    # gaps (V >= H), to reach snow / vegetation / general branches.
    base = rng.uniform(170, 310, size=n_struct)
    tb = np.empty((n_struct, 7))
    tb[:, 0] = base + noise(n_struct, -3, 12)        # 19V
    tb[:, 1] = tb[:, 0] - noise(n_struct, 0, 25)     # 19H
    tb[:, 2] = base + noise(n_struct, -2, 14)        # 22V
    tb[:, 3] = base + noise(n_struct, -8, 8)         # 37V
    tb[:, 4] = tb[:, 3] - noise(n_struct, 0, 20)     # 37H
    tb[:, 5] = base + noise(n_struct, -25, 5)        # 85V
    tb[:, 6] = tb[:, 5] - noise(n_struct, 0, 18)     # 85H
    struct = kelvin_to_packed(tb)

    # Wet-surface targeted: 85V just above 37V (F64 > 0) and 22V near 19V, to
    # drive cells into the deep TEST 25-29 wet-surface retrieval near the
    # WET = 0 decision boundary (the most error-prone branch).
    wbase = rng.uniform(250, 300, size=n_wet)
    tw = np.empty((n_wet, 7))
    tw[:, 0] = wbase + noise(n_wet, -2, 8)           # 19V
    tw[:, 1] = tw[:, 0] - noise(n_wet, 0, 12)        # 19H
    tw[:, 2] = tw[:, 0] + noise(n_wet, -1, 6)        # 22V (F31 small)
    tw[:, 3] = wbase - noise(n_wet, 0, 10)           # 37V
    tw[:, 4] = tw[:, 3] - noise(n_wet, 0, 10)        # 37H
    tw[:, 5] = tw[:, 3] + noise(n_wet, 0, 12)        # 85V (F64 > 0)
    tw[:, 6] = tw[:, 5] - noise(n_wet, 0, 12)        # 85H
    wet = kelvin_to_packed(tw)

    # Raw uniform over the full byte domain (exercises filters / extremes).
    unif = rng.integers(0, 256, size=(n_unif, 7), dtype=np.int64)

    sample = np.concatenate([struct, wet, unif], axis=0)
    rng.shuffle(sample, axis=0)
    return sample.astype(np.int64)


def _float_equal(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    both_nan = np.isnan(a) & np.isnan(b)
    return (a == b) | both_nan


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 12345
    print(f"sample: {n:,} cells  seed={seed}")

    chan = make_sample(n, seed)
    c = core_c.evaluate_packed(chan)
    p = core_numpy.evaluate_packed(chan)

    snow_ok = c.snow == p.snow
    ret_ok = c.ret == p.ret
    temp_ok = _float_equal(c.temp, p.temp)
    wet_ok = _float_equal(c.wet, p.wet)
    all_ok = snow_ok & ret_ok & temp_ok & wet_ok

    n_bad = int((~all_ok).sum())
    print(f"  SNOW  exact: {snow_ok.mean()*100:.4f}%   mism={int((~snow_ok).sum()):,}")
    print(f"  ret   exact: {ret_ok.mean()*100:.4f}%   mism={int((~ret_ok).sum()):,}")
    print(f"  RTEMP exact: {temp_ok.mean()*100:.4f}%   mism={int((~temp_ok).sum()):,}")
    print(f"  WET   exact: {wet_ok.mean()*100:.4f}%   mism={int((~wet_ok).sum()):,}")
    print(f"  TOTAL mismatching cells: {n_bad:,} / {n:,}")

    if n_bad:
        idx = np.flatnonzero(~all_ok)[:12]
        print("\nfirst mismatches (packed channels -> C vs NumPy):")
        hdr = " ".join(f"{nm:>4}" for nm in CHANNEL_NAMES)
        print(f"  {hdr}    engine   RTEMP     WET    SNOW  ret")
        for i in idx:
            vals = " ".join(f"{v:4d}" for v in chan[i])
            print(f"  {vals}    C      {c.temp[i]:8.3f} {c.wet[i]:7.2f} "
                  f"{c.snow[i]:5d} {c.ret[i]:4d}")
            print(f"  {' '*len(vals)}    NumPy  {p.temp[i]:8.3f} {p.wet[i]:7.2f} "
                  f"{p.snow[i]:5d} {p.ret[i]:4d}")
        return 1

    print("\nPERFECT MATCH: NumPy port reproduces the C oracle on every cell.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
