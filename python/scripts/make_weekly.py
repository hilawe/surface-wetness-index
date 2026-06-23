"""Composite a week of CSU daily files into a weekly Surface Wetness product.

    python -m scripts.make_weekly OUT.nc [--calib CAL.json] DAY1.nc ... DAY7.nc

The weekly cadence is the original Basist product cadence. Same engine and
accumulator as the monthly compositor; only the period label differs.
"""

import datetime
import os
import re
import sys

import numpy as np

from swi import monthly, product, core_numpy


class _CalibEngine:
    def __init__(self, coeffs):
        self.coeffs = coeffs

    def evaluate_kelvin(self, tb):
        if self.coeffs is not None:
            from swi import calib_8591 as cal
            tb = cal.apply(tb, self.coeffs)
        return core_numpy.evaluate_kelvin(tb)


def _date(fname):
    m = re.search(r"_D(\d{8})", os.path.basename(fname))
    return datetime.date(int(m.group(1)[:4]), int(m.group(1)[4:6]),
                         int(m.group(1)[6:8])) if m else None


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        return 1
    out = args[0]
    calib_path = None
    if "--calib" in args:
        i = args.index("--calib")
        calib_path = args[i + 1]
        args = args[:i] + args[i + 2:]
    files = args[1:]
    if not files:
        print("no input files given")
        return 1

    coeffs = None
    calib_note = "none (91 GHz used as 85 GHz)"
    if calib_path:
        from swi import calib_8591 as cal
        coeffs = cal.load_fit(calib_path)
        calib_note = (f"85-to-91 GHz adjustment applied (model={coeffs['model']}, "
                      f"from {os.path.basename(calib_path)})")

    print(f"compositing {len(files)} day(s)" + (" with calibration" if coeffs else ""))
    comp = monthly.composite(files, engine=_CalibEngine(coeffs))
    print(f"  used: asc={comp['used'].get('asc',0)} dsc={comp['used'].get('dsc',0)}  "
          f"skipped: {comp['skipped']}")

    base = os.path.basename(files[0])
    sat = (re.search(r"_(F\d{2}|GCOMW1)_", base) or [None, ""])[1]
    dates = sorted(d for d in (_date(f) for f in files) if d)
    if dates:
        iso = dates[0].isocalendar()
        label = f"{iso[0]}-W{iso[1]:02d}"
        tcs, tce = dates[0].isoformat(), dates[-1].isoformat()
    else:
        label, tcs, tce = "", "", ""
    meta = {"sensor": sat, "label": label, "n_days": len(files),
            "time_coverage_start": tcs, "time_coverage_end": tce,
            "source": f"{len(files)} CSU daily files",
            "grid_resolution": "0.25 degree", "calibration": calib_note}
    product.write_weekly_product(out, comp["lat"], comp["lon"], comp["by_pass"], meta)
    print(f"wrote {out}  (sensor={sat}, week={label}, n_days={len(files)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
