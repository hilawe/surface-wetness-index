"""Which of the decision tree's conditions can actually be reached?

The manuscript claimed the regression sample was "constructed to exercise every
branch of the tree". That claim was never measured, and it is false: at least three
conditions cannot be satisfied by any input, because their own path constraints
contradict them. This driver replaces the claim with a measurement.

It pushes a large random sample through the instrumented evaluator, which counts how
many cells satisfy each of the tree's conditions, and reports three groups:

  reached        the condition is true for some input, so the branch is live
  never true     no input in the sample satisfied it, which is either a rare branch
                 or an unreachable one, and the two are distinguished below
  unreachable    proven so by the path constraints, independent of sampling

The unreachable set is not a defect in the port. The port reproduces the recovered
2004 C source exactly, including its dead code, and the regression test enforces
that. It is a property of the original operational algorithm, and it is reported
here as part of the archaeology rather than quietly fixed.

Usage:
    python -m scripts.branch_coverage [--cells 5000000] [--seed 12345]
        [--chunk 500000] [--out J.json]
"""

import json
import os
import sys

import numpy as np

from swi import core_numpy
from swi.channels import N_CHANNELS

# Conditions provable unreachable from the path constraints, with the proof. These
# are statements about the recovered 2004 source, checked by hand against
# src/sig_recog.c, not guesses from the sampling.
PROVEN_UNREACHABLE = {
    "c28": (
        "TEST 28 requires F46 > 0 inside the TEST 25 block, which requires F64 > 0. "
        "F46 = P4 - P6 and F64 = P6 - P4, so F46 = -F64 identically and the two "
        "cannot both be positive."
    ),
    "c29": (
        "TEST 29 is nested inside TEST 28, which is unreachable, so it cannot be "
        "evaluated regardless of its own condition."
    ),
    "c28else": (
        "The fall-through limb inside TEST 28's satisfied branch, taken when "
        "TEST 28 holds and TEST 29 does not. TEST 28's condition can never "
        "hold, so neither TEST 29 nor this limb "
        "can be taken."
    ),
    "c23": (
        "TEST 23 asks whether RTEMP < 271 K, where RTEMP = 1.0714 P3 + 0.2183 F36. "
        "Reaching TEST 21 with CONTAM > 0 requires surviving TEST 10, which returns "
        "whenever CONTAM > 0 and P1 < 261, so P1 >= 261 K. TEST 22 requires "
        "F31 = P3 - P1 >= -3, hence P3 >= 258 K, and TEST 21 requires F36 > 0. "
        "Therefore RTEMP > 1.0714 x 258 = 276.4 K at TEST 23 and the condition can "
        "never hold."
    ),
}
# Conditions where one clause of an OR is dead but the condition as a whole is live.
PARTIALLY_DEAD = {
    "c15": (
        "TEST 15 is reached only through TEST 13, which requires P3 < 257. Its first "
        "clause, P3 >= 258, is therefore never satisfied. The other two clauses are "
        "reachable, so the condition as a whole is live."
    ),
}


def opt(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def main():
    argv = sys.argv
    cells = int(opt(argv, "--cells", "5000000"))
    seed = int(opt(argv, "--seed", "12345"))
    chunk = int(opt(argv, "--chunk", "500000"))
    out_path = opt(argv, "--out", "../results/branch_coverage.json")

    rng = np.random.RandomState(seed)
    trace = {}
    done = 0
    while done < cells:
        n = min(chunk, cells - done)
        # Packed domain is Kelvin - 70. Sample widely enough to reach the cold
        # scattering and warm wet-surface regimes the tree separates, and include
        # the sentinel-adjacent low end so the rejection paths are exercised.
        block = rng.randint(0, 256, size=(n, N_CHANNELS)).astype(np.int64)
        core_numpy.evaluate_packed(block, trace=trace)
        done += n

    names = sorted(trace, key=lambda k: (int("".join(ch for ch in k if ch.isdigit())), k))
    reached = {k: trace[k] for k in names if trace[k] > 0}
    never = [k for k in names if trace[k] == 0]

    res = {
        "cells": cells, "seed": seed,
        "n_conditions": len(names),
        "condition_semantics": (
            "Each label counts inputs for which the named condition HOLDS. "
            "cNN is a numbered test's condition, cNNelse the complementary limb "
            "actually taken in the code (for c28else, the limb inside TEST 28's "
            "satisfied branch when TEST 29 does not hold), and c26_live and "
            "c26_noc28 are diagnostic sub-populations of TEST 26. The 42 "
            "instrumented conditions are not a one-to-one relabeling of the 42 "
            "numbered tests."),
        "uninstrumented_tests": (
            "TESTS 1 to 4 are the gap and bad-data screens applied to every "
            "cell before the instrumented region; TESTS 11, 12, 39, and 40 are "
            "unconditional assignments, not conditions."),
        "reached": reached,
        "never_true_in_sample": never,
        "proven_unreachable": {k: v for k, v in PROVEN_UNREACHABLE.items() if k in names},
        "partially_dead": {k: v for k, v in PARTIALLY_DEAD.items() if k in names},
    }

    print(f"\nBranch coverage over {cells:,} random cells (seed {seed})")
    print(f"  conditions instrumented : {len(names)}")
    print(f"  reached at least once   : {len(reached)}")
    print(f"  never true in sample    : {len(never)}"
          + (f"  {never}" if never else ""))

    unproven = [k for k in never if k not in PROVEN_UNREACHABLE]
    print("\n  proven unreachable by path constraints:")
    for k, why in PROVEN_UNREACHABLE.items():
        mark = "confirmed, never true in sample" if k in never else \
               "WARNING: proof says unreachable but the sample reached it"
        print(f"    {k}: {mark}")
        print(f"        {why}")
    if unproven:
        print("\n  never true in sample but NOT proven unreachable "
              "(rare, or needs a targeted case):")
        for k in unproven:
            print(f"    {k}")
    print("\n  conditions with a dead clause but a live condition:")
    for k, why in PARTIALLY_DEAD.items():
        print(f"    {k}: {why}")

    res["unproven_never_true"] = unproven
    ok = all(k in never for k in PROVEN_UNREACHABLE)
    res["proofs_consistent_with_sample"] = bool(ok)
    print(f"\n  proofs consistent with the sample: {ok}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
