"""Small committed caches of exactly the arrays a manuscript figure draws.

A figure that can only be rebuilt from large raw inputs (satellite grids,
reference datasets) stops being reproducible the moment those inputs are
deleted, lost in a restore, or removed upstream. The inputs to this project's
validation figures live in untracked ``data/`` and ``scratch/`` paths, so a
destructive operation or a partial restore erases the ability to regenerate a
figure with no record of what was lost.

This module closes that gap. When a figure is built from raw inputs the script
also saves the handful of arrays it actually plots into a compressed ``.npz``
plus a human-readable ``.json`` provenance sidecar, both small enough to commit.
The figure can then be redrawn from that cache alone, with ``--from-cache``,
independent of the raw inputs and even of the upstream archives.

The cache is a reproducibility backstop, not the source of truth: the scripts
still rebuild from raw inputs when they are available, and a real rebuild
refreshes the cache so it cannot drift from the plotting code.
"""

import hashlib
import json
import os

import numpy as np

# Committed caches live under the repository, resolved from this module's own
# location so the path does not depend on a script's cwd or its --out argument.
_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "results", "figure_cache"))


def cache_path(figure):
    """Return the committed cache path for a figure name (e.g. 'fig_product')."""
    return os.path.join(_CACHE_DIR, figure + ".npz")


def sha256(path):
    """Return the SHA256 hex digest of a file, for provenance manifests."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def as_stored(**arrays):
    """Return the arrays cast exactly as :func:`save` will store them.

    Float arrays become float32 (a figure is a visualization, not a computation,
    so float32 is lossless at plotted precision); other dtypes, including boolean
    masks and integers, are left unchanged. A raw build plots THESE arrays, not
    the original float64 ones, so a --from-cache redraw is identical to the raw
    build rather than merely close: both plot the same float32 values.
    """
    out = {}
    for name, value in arrays.items():
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
        out[name] = arr
    return out


def save(npz_path, meta, **arrays):
    """Write the plotting arrays (as :func:`as_stored` casts them) and a JSON
    provenance sidecar. ``meta`` records how the figure was built and is written
    both inside the npz and as a sibling ``.json`` so it is greppable without
    loading numpy. Call this only after the figure has rendered successfully, so
    a failed render cannot leave a fresh cache beside a stale figure.
    """
    os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
    payload = dict(as_stored(**arrays))
    payload["_meta_json"] = np.frombuffer(
        json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(npz_path, **payload)
    with open(os.path.splitext(npz_path)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print(f"wrote figure cache {npz_path}")


def load(npz_path):
    """Return (arrays dict, meta dict) from a cache written by :func:`save`.

    ``allow_pickle`` stays False, so a cache file cannot execute code on load.
    """
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(bytes(z["_meta_json"]).decode("utf-8"))
    arrays = {k: z[k] for k in z.files if k != "_meta_json"}
    return arrays, meta
