"""Build monthly Surface Wetness products for many months at once.

    python -m scripts.build_monthly_multiyear OUTDIR [--calib CAL.json] DATADIR ...

Scans each DATADIR for CSU daily F-16 files, groups them by calendar month, and
writes one calibrated monthly product per month to
OUTDIR/SWI_F16_<YYYYMM>_monthly.nc. A month whose product already exists is
skipped, so the driver is restartable. This builds the multi-year record for the
temporal anomaly validation without ad hoc shell loops.
"""

import glob
import os
import re
import sys

from swi import core_numpy, monthly, product


class _CalibEngine:
    def __init__(self, coeffs):
        self.coeffs = coeffs

    def evaluate_kelvin(self, tb):
        if self.coeffs is not None:
            from swi import calib_8591 as cal
            tb = cal.apply(tb, self.coeffs)
        return core_numpy.evaluate_kelvin(tb)


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        return 1
    outdir = args[0]
    calib_path = None
    if "--calib" in args:
        i = args.index("--calib")
        calib_path = args[i + 1]
        args = args[:i] + args[i + 2:]
    datadirs = args[1:]
    if not datadirs:
        print("no data directories given")
        return 1
    os.makedirs(outdir, exist_ok=True)

    coeffs = None
    calib_note = "none (91 GHz used as 85 GHz)"
    if calib_path:
        from swi import calib_8591 as cal
        coeffs = cal.load_fit(calib_path)
        calib_note = (f"85-to-91 GHz adjustment applied (model={coeffs['model']}, "
                      f"from {os.path.basename(calib_path)})")

    months = {}
    for d in datadirs:
        for f in glob.glob(os.path.join(d, "CSU_*F16_D*.nc")):
            m = re.search(r"_D(\d{6})\d{2}", os.path.basename(f))
            if m:
                months.setdefault(m.group(1), []).append(f)

    engine = _CalibEngine(coeffs)
    for ym in sorted(months):
        out = os.path.join(outdir, f"SWI_F16_{ym}_monthly.nc")
        if os.path.exists(out):
            print(f"{ym}: exists, skip")
            continue
        files = sorted(months[ym])
        comp = monthly.composite(files, engine=engine)
        meta = {"sensor": "F16", "month": ym, "n_days": len(files),
                "source": f"{len(files)} CSU daily files",
                "grid_resolution": "0.25 degree", "calibration": calib_note}
        product.write_monthly_product(
            out, comp["lat"], comp["lon"], comp["by_pass"], meta)
        print(f"{ym}: wrote {os.path.basename(out)} (n_days={len(files)}, "
              f"used asc={comp['used'].get('asc', 0)} dsc={comp['used'].get('dsc', 0)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
