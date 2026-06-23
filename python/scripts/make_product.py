"""Produce a daily Surface Wetness Index NetCDF from a CSU ICDR-GRID file.

    python -m scripts.make_product CSU_..._F16_D20260522.nc out.nc

Runs both passes through the engine (raw 91-as-85 for v0; calibration is applied
once the 85-to-91 fit is available) and writes a CF/ACDD product file.
"""

import os
import re
import sys

import numpy as np

from swi import io_csu_grid as io, product, core_numpy


class _CalibEngine:
    """Engine wrapper that applies an 85-to-91 adjustment before evaluating.

    Exposes evaluate_kelvin(tb) so it is a drop-in for io.evaluate_file's engine
    argument. With coeffs=None it is a passthrough to core_numpy.
    """

    def __init__(self, coeffs):
        self.coeffs = coeffs

    def evaluate_kelvin(self, tb):
        if self.coeffs is not None:
            from swi import calib_8591 as cal
            tb = cal.apply(tb, self.coeffs)
        return core_numpy.evaluate_kelvin(tb)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    src, out = sys.argv[1], sys.argv[2]
    calib_path = None
    if "--calib" in sys.argv:
        calib_path = sys.argv[sys.argv.index("--calib") + 1]

    coeffs = None
    calib_note = "none (91 GHz used as 85 GHz; 85-to-91 fit not yet applied)"
    if calib_path:
        from swi import calib_8591 as cal
        coeffs = cal.load_fit(calib_path)
        calib_note = (f"85-to-91 GHz adjustment applied (model={coeffs['model']}, "
                      f"from {os.path.basename(calib_path)})")

    by_pass = {}
    sensor = ""
    for p in ("asc", "dsc"):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = io.evaluate_file(src, pass_=p, engine=_CalibEngine(coeffs))
        sensor = r["sensor"]
        if r.get("empty_channels"):
            print(f"WARNING: {p} pass missing channel(s) {r['empty_channels']}; "
                  f"that pass will be all-fill.")
        by_pass[p] = r
        lat, lon = r["lat"], r["lon"]

    m = re.search(r"_D(\d{8})", os.path.basename(src))
    meta = {
        "sensor": sensor,
        "date": m.group(1) if m else "",
        "source": os.path.basename(src),
        "grid_resolution": "0.25 degree",
        "calibration": calib_note,
    }
    product.write_daily_product(out, lat, lon, by_pass, meta)
    print(f"wrote {out}  (sensor={sensor}, date={meta['date']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
