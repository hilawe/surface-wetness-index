"""ctypes binding to the original Basist C engine (the reference oracle).

Loads libswi (built from src/ via `make`) and exposes a vectorized-looking
``evaluate`` that loops over cells inside C. Bit-for-bit faithful to the 2004
source, used to validate the NumPy port.
"""

import ctypes
import os
import sys

import numpy as np

from .channels import N_CHANNELS, kelvin_to_packed
from .core_numpy import Result  # the engine result namedtuple (defined there)

_LIBNAMES = ["libswi.dylib", "libswi.so"]


def _find_lib():
    # 1) explicit override
    env = os.environ.get("SWI_LIB")
    if env and os.path.exists(env):
        return env
    # 2) alongside the source tree: <repo>/src/libswi.*
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.normpath(os.path.join(here, "..", "..", "src"))
    for name in _LIBNAMES:
        cand = os.path.join(src, name)
        if os.path.exists(cand):
            return cand
    # 3) current working directory
    for name in _LIBNAMES:
        if os.path.exists(name):
            return os.path.abspath(name)
    raise FileNotFoundError(
        "libswi not found. Build it first:  (cd src && make)  "
        "or set the SWI_LIB environment variable to its path."
    )


def _load():
    lib = ctypes.CDLL(_find_lib())
    f = lib.swi_eval_packed
    f.restype = None
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    f.argtypes = [i32, ctypes.c_int, f32, f32, i32, i32]
    return lib, f


_LIB, _EVAL = _load()


def evaluate_packed(chan):
    """Evaluate native packed integer channels.

    chan : array-like, shape (..., 7), integer packed values (Kelvin - 70).
    returns Result of arrays shaped like chan.shape[:-1] (temp/wet float32,
    snow/ret int32).
    """
    chan = np.ascontiguousarray(chan, dtype=np.int32)
    if chan.shape[-1] != N_CHANNELS:
        raise ValueError(f"last axis must be {N_CHANNELS} channels, got {chan.shape}")
    out_shape = chan.shape[:-1]
    n = int(np.prod(out_shape)) if out_shape else 1
    flat = chan.reshape(n, N_CHANNELS)

    temp = np.empty(n, dtype=np.float32)
    wet = np.empty(n, dtype=np.float32)
    snow = np.empty(n, dtype=np.int32)
    ret = np.empty(n, dtype=np.int32)

    _EVAL(np.ascontiguousarray(flat), n, temp, wet, snow, ret)

    return Result(
        temp.reshape(out_shape),
        wet.reshape(out_shape),
        snow.reshape(out_shape),
        ret.reshape(out_shape),
    )


def evaluate_kelvin(tb):
    """Evaluate Kelvin brightness temperatures (rounded to the packed domain)."""
    return evaluate_packed(kelvin_to_packed(tb))


if __name__ == "__main__":  # quick parity check with src/swi_smoke.c
    sample = np.array([
        [32, 32, 32, 32, 32, 32, 32],
        [200, 185, 202, 198, 185, 195, 188],
        [150, 148, 160, 175, 170, 185, 178],
    ], dtype=np.int32)
    r = evaluate_packed(sample)
    for i in range(3):
        print(f"cell {i}: RTEMP={r.temp[i]:8.3f}  WET={r.wet[i]:7.2f}  "
              f"SNOW={r.snow[i]:4d}  ret={r.ret[i]:3d}")
    sys.exit(0)
