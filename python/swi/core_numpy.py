"""Vectorized NumPy port of the Basist signal-recognition decision tree.

Faithful reimplementation of ``sig_recog`` (src/sig_recog.c). Ported test-by-test
from the C source so it matches the C oracle (core_c) cell-by-cell. The early
``return`` structure of the tree is reproduced with a boolean ``done`` mask:
once a cell is finalized it is never touched again, so test order is preserved.

Precision rules chosen to mirror the C exactly:
- Channel values and all polarization/scattering differences are integers.
- Stored regressions (RTEMP, WET, the wet-surface P) are computed in float64
  then cast to float32, because the C globals are 32-bit ``float``.
- Pure comparison expressions that the C evaluates in ``double`` without an
  intermediate float store (Tests 14 and 15) are done in float64.
- Test 6's ``PD19/10`` is float32 division, as in the C (PD19 is ``float``).

Channel order: 19V 19H 22V 37V 37H 85V 85H. Input is the native packed domain
(packed = Kelvin - 70); the +70 offset is applied here as in sig_recog.
"""

from collections import namedtuple

import numpy as np

from .channels import N_CHANNELS, NV

# The engine result. Defined here (not in core_c) so that importing the NumPy
# engine never requires the compiled C oracle (libswi); core_c reuses this.
Result = namedtuple("Result", ["temp", "wet", "snow", "ret"])

RAIN_WET = 1
WET_SURF = 2

_f32 = np.float32


def _wet_surf(P1, P5, P6, F31, F64):
    """gen_WET_surf: returns (P_float32, WET_float32) over whole arrays.

    Depends only on constant channel arrays, so it is computed once.
    P is stored to a 32-bit float in C, then reused (promoted) in the WET
    expression; we replicate that intermediate rounding.
    """
    P = 0.0783 * F31 - 0.4369 * P5 + 1.4828 * P6 - 0.6628 * F64  # float64
    Pf = P.astype(_f32)
    with np.errstate(divide="ignore", invalid="ignore"):
        wet = ((1.0 - (P1.astype(np.float64) / (Pf.astype(np.float64) / 1.06)))
               / 0.33) * 100.0
    return Pf, wet.astype(_f32)


def evaluate_packed(chan):
    """Evaluate native packed integer channels (Kelvin - 70), shape (..., 7)."""
    chan = np.asarray(chan, dtype=np.int64)
    if chan.shape[-1] != N_CHANNELS:
        raise ValueError(f"last axis must be {N_CHANNELS} channels, got {chan.shape}")
    out_shape = chan.shape[:-1]
    flat = chan.reshape(-1, N_CHANNELS)
    n = flat.shape[0]

    p1 = flat[:, 0]  # raw packed (pre-offset) values

    # Outputs and per-cell state.
    rtemp = np.zeros(n, dtype=_f32)
    wet = np.zeros(n, dtype=_f32)
    snow = np.zeros(n, dtype=np.int32)
    ret = np.zeros(n, dtype=np.int32)
    done = np.zeros(n, dtype=bool)

    def fin(mask, rt=None, wt=None, sn=None, rc=None):
        """Finalize cells in mask: assign outputs, mark done."""
        if rt is not None:
            rtemp[mask] = rt[mask] if isinstance(rt, np.ndarray) else rt
        if wt is not None:
            wet[mask] = wt[mask] if isinstance(wt, np.ndarray) else wt
        if sn is not None:
            snow[mask] = sn[mask] if isinstance(sn, np.ndarray) else sn
        if rc is not None:
            ret[mask] = rc
        done[mask] = True

    # TEST 1: orbital gap (checked on the raw pre-offset 19V).
    fin(p1 < 100, rt=_f32(-99.9), wt=_f32(-99.9), sn=-100, rc=-1)

    # Apply the +NV offset -> Kelvin domain (constant hereafter).
    P1 = flat[:, 0] + NV
    P2 = flat[:, 1] + NV
    P3 = flat[:, 2] + NV
    P4 = flat[:, 3] + NV
    P5 = flat[:, 4] + NV
    P6 = flat[:, 5] + NV
    P7 = flat[:, 6] + NV

    PD19 = P1 - P2
    PD37 = P4 - P5
    PD85 = P6 - P7
    F13 = P1 - P3
    F14 = P1 - P4
    F36 = P3 - P6
    F31 = P3 - P1
    F34 = P3 - P4
    F41 = P4 - P1
    F43 = P4 - P3
    F46 = P4 - P6
    F64 = P6 - P4

    # Constant wet-surface retrieval (gen_WET_surf), used in Tests 5/6 and 25.
    Pglob, WETsurf = _wet_surf(P1, P5, P6, F31, F64)

    # Default: vegetated land.
    live = ~done
    rtemp[live] = (1.0650 * P3[live]).astype(_f32)
    wet[live] = 0.0
    snow[live] = 0
    nop = np.ones(n, dtype=np.int32)
    wets = np.zeros(n, dtype=np.int32)
    contam = np.zeros(n, dtype=np.int32)
    lime = np.zeros(n, dtype=np.int32)

    # TEST 2
    live = ~done
    c = live & ((PD19 < -1) | (PD37 < -1) | (PD85 < -1) | (PD19 > 45))
    fin(c, rt=_f32(-99), wt=_f32(-99), sn=-99, rc=-1)

    # TEST 3
    live = ~done
    c = live & ((F64 > 50) | (F64 < -50) | (F13 > 30) | (F13 < -30) | (F34 > 50))
    fin(c, rt=_f32(-99), wt=_f32(-99), sn=-99, rc=-1)

    # TEST 4
    live = ~done
    c = live & ((P3 <= 210) | ((P3 <= 229) & (P6 <= 240) & ((F36 < 0) | (F46 < 0))))
    fin(c, rt=_f32(-99), wt=_f32(0.0), sn=-1, rc=-1)

    # TEST 5 / 6
    live = ~done
    c5 = live & (F31 >= 1) & (PD85 >= 15) & (P4 > P3)
    pd19_div = (PD19.astype(_f32) / _f32(10))            # float32 division (Test 6)
    c6 = c5 & (((pd19_div <= F31.astype(_f32)) | (PD19 < 25)) & (P6 > P1))
    # cond6 cells take the gen_WET_surf value; the rest of c5 get WET = -99.
    wet[c6] = WETsurf[c6]
    wet[c5 & ~c6] = _f32(-99.0)
    snow[c5] = 0
    rtemp[c5] = _f32(-99.0)
    ret[c5] = -1
    done[c5] = True

    # TEST 7 block (CONTAM accumulation + TEST 10)
    live = ~done
    c7 = live & (PD37 <= 7)
    c8 = c7 & (F46 > 3) & (P1 >= 257) & (PD85 <= 7)
    contam[c8] = F46[c8]
    c9 = c7 & ~c8 & (F31 > 0) & (F43 > 0) & (F64 <= 0)
    contam[c9] = F43[c9]
    c10 = c7 & ((contam > 25) | ((contam > 0) & (P1 < 261)))
    fin(c10, rt=_f32(-99.0), wt=_f32(0.0), sn=0, rc=1)

    # SNOW FILTER
    scat = F36.copy()
    scat = np.where(F14 > scat, F14, scat)               # TEST 11
    scat = np.where(F46 > scat, F46, scat)               # TEST 12
    live = ~done
    snow[live] = 0
    c13 = live & (scat >= 1) & (P3 < 257)                # TEST 13
    snow[c13] = scat[c13]
    # TEST 14
    c14 = c13 & (PD85.astype(np.float64) >= 2.5 * snow.astype(np.float64))
    snow[c14] = 0
    # TEST 15
    c15 = c13 & ((P3 >= 258)
                 | (P3.astype(np.float64) >= 165.0 + 0.49 * P6.astype(np.float64))
                 | ((P3 >= 254) & (scat <= 2) & (PD85 >= 3)))
    snow[c15] = 0
    # TEST 16 / 17
    c16 = c13 & (PD19 >= 18) & (F14 <= 10) & (F46 <= 5) & (F31 <= 0) & (P3 > 235)
    c17 = c16 & (P3 > 235) & (PD37 < 30)
    snow[c17] = 0
    c16else = c16 & ~c17
    fin(c16else, rt=_f32(-99.0), wt=_f32(-99.0), sn=0, rc=1)
    # TEST 18
    live = ~done
    c18 = live & c13 & (snow != 0) & (PD19 >= 12) & (scat <= 2) & (F14 <= 2)
    fin(c18, rt=_f32(-99.0), wt=_f32(-99.0), sn=-99, rc=1)
    # TEST 19  (note C precedence: F34>17 || (P6>P4 && P6>245))
    live = ~done
    c19 = (live & c13
           & ((F34 > 17) | ((P6 > P4) & (P6 > 245)))
           & ((F31 > 10) | (PD85 > 20)))
    fin(c19, rt=_f32(-99.0), wt=_f32(-99.0), sn=-99, rc=1)

    # TEST 20: remove remaining snow
    live = ~done
    c20 = live & (snow > 0)
    fin(c20, rt=_f32(-99.0), wt=_f32(0.0), rc=1)

    # TEST 21 / 22 / 23: rain or snow over a wet surface
    live = ~done
    c21 = live & (contam > 0) & (F36 > 0) & (PD37 < 7)
    c22 = c21 & (F31 >= -3) & (PD85 <= 10)
    wets[c22] = RAIN_WET
    rtemp[c22] = (1.0714 * P3[c22] + 0.2183 * F36[c22]).astype(_f32)
    c23 = c22 & (rtemp < _f32(271.0))
    fin(c23, rt=_f32(-99.0), sn=F36, wt=_f32(0.0), rc=1)
    c22else = c22 & ~c23
    nop[c22else] = 0
    fin(c22else, wt=_f32(-99.0), rc=1)            # RTEMP keeps the regression value
    c21else = c21 & ~c22
    fin(c21else, rt=_f32(-99.0), sn=0, wt=_f32(-99.0), rc=1)

    # TEST 24: rain
    live = ~done
    c24 = live & (((F46 > 5) & (P3 >= 257) & (PD85 <= 5) & (PD37 < 7))
                  | ((P3 >= 257) & (F46 > 10) & (PD37 < 7)))
    fin(c24, rt=_f32(-99.0), wt=_f32(-99.0), sn=-99, rc=1)

    # TEST 25 block: a wet surface
    live = ~done
    c25 = (live
           & ((F31 > 3) | (F64 * 2 >= PD37) | (F41 * 7 > PD19))
           & (contam == 0) & (snow == 0) & (F64 > 0))
    wet[c25] = WETsurf[c25]                       # gen_WET_surf
    # Snapshot the WET>0 split NOW, before any finalize below mutates wet.
    c26 = c25 & (wet > 0)                          # TEST 26 (true branch)
    c26else = c25 & ~(wet > 0)                     # TEST 26 (else: WET <= 0)
    # TEST 27
    c27 = c26 & (PD19 > F64 * 4) & (F64 >= 5)
    fin(c27, rt=_f32(-99.0), wt=_f32(-99.0), sn=-99, rc=1)
    # TEST 28 / 29
    live = ~done
    c28 = c26 & live & (F34 > 0) & (F46 > 0)
    c29 = c28 & ((PD19 - F46) >= 8)
    fin(c29, rt=_f32(-99.0), sn=0, wt=_f32(0.0), rc=1)
    c28else = c28 & ~c29
    rtemp[c28else] = (0.3204 * PD19[c28else] + 1.0558 * P3[c28else]
                      - 0.5008 * F46[c28else]).astype(_f32)
    # else branch of "if(F34>0 && F46>0)": RTEMP = P
    c26_noc28 = c26 & ~done & ~c28
    rtemp[c26_noc28] = Pglob[c26_noc28]
    # after the inner if/else, for all still-live c26 cells: nop=0, wets=WET_SURF
    c26_live = c26 & ~done
    nop[c26_live] = 0
    wets[c26_live] = WET_SURF
    # else branch of TEST 26 (WET <= 0)
    rtemp[c26else] = (0.5195 * P1[c26else] + 1.0869 * P3[c26else]
                      - 0.5375 * P4[c26else]).astype(_f32)
    fin(c26else, wt=_f32(0.0), sn=0, rc=1)

    # TEST 30: glacial
    live = ~done
    c30 = live & (wet != 0.0) & (rtemp <= _f32(258.0)) & (P6 < 256)
    fin(c30, sn=-1, wt=_f32(-99.0), rt=_f32(-99.0), rc=1)

    # TEST 31: quartz (no return)
    live = ~done
    c31 = live & (P1 >= P3) & (PD19 > 25) & (P3 + 2 <= P4) & (P4 > P6)
    nop[c31] = 0

    # TEST 32: limestone
    live = ~done
    c32 = live & (wets == 0) & (P4 < P6) & (PD37 > 6)
    nop[c32] = 0
    lime[c32] = 1
    rtemp[c32] = (0.31091196 * PD19[c32] + 0.56659491 * P4[c32]
                  + 0.47783562 * P6[c32]).astype(_f32)
    fin(c32, wt=_f32(0.0), sn=0, rc=1)

    # TEST 33
    live = ~done
    c33 = live & (((PD19 > 20) & (PD19 + 2 < PD37))
                  | ((PD19 > 10) & (PD19 + 4 < PD37) & (PD85 > PD37)))
    fin(c33, rt=_f32(-99.0), wt=_f32(-99.0), sn=0, rc=1)

    # TEST 34
    live = ~done
    c34 = live & (P1 <= 256) & (F36 <= -4) & (wet <= 0.0) & (PD85 > 5)
    fin(c34, sn=-99, wt=_f32(-99.0), rt=_f32(-99.0), rc=1)

    # TEST 35 / 36
    live = ~done
    c35 = live & (wet <= 0.0) & (PD19 > 8) & ((PD85 > PD37) | (PD85 > PD19))
    c36 = c35 & ((np.abs(PD19) + np.abs(PD37) + np.abs(PD85)) >= 5)
    fin(c36, rt=_f32(-99.0), wt=_f32(-99.0), sn=-99, rc=-1)

    # TEST 37
    live = ~done
    c37 = live & (((PD37 > 39) & (wet > 0))
                  | ((PD85 > 20) & (wet == 0) & (PD37 < 30))
                  | ((wets == WET_SURF) & (F36 > 0))
                  | ((PD19 > 10) & (wet == 0) & (F64 > 3) & (lime == 0)))
    fin(c37, rt=_f32(-99.0), wt=_f32(-99.0), sn=-99, rc=1)

    # TEST 38 (no return)
    live = ~done
    c38 = live & (P3 > P1) & (P3 > P6)
    rtemp[c38] = (1.0730 * P3[c38] + 0.2260 * F36[c38]).astype(_f32)
    wet[c38] = _f32(-99.0)
    snow[c38] = 0
    nop[c38] = 0

    # TEST 39
    live = ~done
    wet[live & (wet < 0.0)] = _f32(0.0)

    # TEST 40
    live = ~done
    fin(live & (rtemp == _f32(-99)), rc=-1)

    # TEST 41 / 42: vegetation is last
    live = ~done
    c41 = live & (nop == 1)
    c42 = c41 & (F31 > 0) & (PD37 > 10)
    fin(c42, rt=_f32(-99.0), rc=1)
    c42else = c41 & ~c42
    rtemp[c42else] = (1.0698 * P3[c42else]).astype(_f32)
    # remaining live cells return 0 (ret already 0) with current RTEMP/WET/SNOW.

    return Result(
        rtemp.reshape(out_shape),
        wet.reshape(out_shape),
        snow.reshape(out_shape),
        ret.reshape(out_shape),
    )


def evaluate_kelvin(tb):
    """Evaluate Kelvin brightness temperatures (rounded to the packed domain)."""
    from .channels import kelvin_to_packed
    return evaluate_packed(kelvin_to_packed(tb))
