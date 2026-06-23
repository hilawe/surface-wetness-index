"""Surface Wetness Index (Basist) engine: C oracle and NumPy port.

- ``core_c``      : ctypes binding to the original 2004 C decision tree (oracle).
- ``core_numpy``  : vectorized NumPy reimplementation, validated against the oracle.
- ``channels``    : channel order, packing offset, and output sentinels.
"""

from . import channels  # noqa: F401

__all__ = ["channels"]
