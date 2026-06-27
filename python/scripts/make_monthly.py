"""Composite a month of CSU daily files into a monthly Surface Wetness product.

    python -m scripts.make_monthly OUT.nc [--calib CAL.json] FILE1.nc FILE2.nc ...

All input files should be the same satellite and month. Runs the engine on each
day (both passes), accumulates monthly statistics, and writes a CF/ACDD monthly
product.
"""

import os
import re
import sys

import numpy as np

from swi import monthly, product, core_numpy, calib_8591




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

    print(f"compositing {len(files)} day(s)" + (f" with calibration" if coeffs else ""))
    comp = monthly.composite(files, engine=calib_8591.make_engine(coeffs))
    print(f"  used: asc={comp['used'].get('asc',0)} dsc={comp['used'].get('dsc',0)}  "
          f"skipped: {comp['skipped']}")

    # derive sensor and month from the first filename
    base = os.path.basename(files[0])
    sat = (re.search(r"_(F\d{2}|GCOMW1)_", base) or [None, ""])[1]
    ym = (re.search(r"_D(\d{6})", base) or [None, ""])[1]
    meta = {"sensor": sat, "month": ym, "n_days": len(files),
            "source": f"{len(files)} CSU daily files",
            "grid_resolution": "0.25 degree", "calibration": calib_note}
    product.write_monthly_product(out, comp["lat"], comp["lon"], comp["by_pass"], meta)
    print(f"wrote {out}  (sensor={sat}, month={ym}, n_days={len(files)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
