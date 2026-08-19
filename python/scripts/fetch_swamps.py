"""Download SWAMPS fractional surface water monthly means from the IRI Data Library.

    python -m scripts.fetch_swamps YYYYMM [YYYYMM ...] [--out DIR] [--force]

SWAMPS (Surface WAter Microwave Product Series) version 3.1, stable record, daily
fractional surface water (0 to 1), from the NASA JPL wetlands collection in the
IRI Data Library (iridl.ldeo.columbia.edu). This is the reference behind the
paper's inundation figure (fig_swamps) and the SSMI-era transfer test. The input
was lost once with no fetcher, so this script is the recovery path.

For each month it asks the IRI Data Library to average the daily field over that
month and returns a NetCDF with variables lat, lon, and fw, which is exactly the
layout scripts.validate_swamps.load_swamps_fw reads. Files are written to
../data/swamps/swamps_fw_YYYYMM.nc and skipped if already present and valid.

The record runs from 1992-02 to 2017-01. Months outside that range are refused.
The IRI Data Library HTML catalog needs a login, but the DODS data endpoint used
here is open, so no credentials are required. A returned file is validated (real
lat/lon/fw arrays, fractional water in range) rather than trusted by size, because
the server can answer an error as a 200.
"""

import calendar
import os
import re
import sys
import urllib.parse
import urllib.request

# The open DODS data endpoint for the SWAMPS v3.1 stable daily fractional-water
# variable. The HTML view of this path redirects to a login; the data endpoints
# (dods, data.nc) do not.
BASE = ("https://iridl.ldeo.columbia.edu/SOURCES/.NASA/.JPL/.wetlands/"
        ".dailyinundation/.swamps_v3p1/.stable/.fw")

# Record coverage, inclusive, as YYYYMM. Feb 1992 to Jan 2017.
FIRST_MONTH, LAST_MONTH = 199202, 201701


def month_mean_url(ym):
    """IRI Data Library URL for the monthly mean of daily fw over month YYYYMM."""
    year, mon = int(ym[:4]), int(ym[4:6])
    last = calendar.monthrange(year, mon)[1]
    abbr = calendar.month_abbr[mon]
    # Select the month by date range on the daily T axis, then average over T,
    # which collapses fw to (lat, lon).
    expr = (f"/T/(1 {abbr} {year})/({last} {abbr} {year})/RANGEEDGES"
            f"/[T]average/data.nc")
    return BASE + urllib.parse.quote(expr, safe="/()")


def validate(path):
    """Return (ok, message). A valid file has real lat/lon/fw with fractional
    water in [0, 1] over some cells, not an HTML error served as a .nc."""
    try:
        import numpy as np
        from netCDF4 import Dataset
    except Exception as exc:  # noqa: BLE001
        return False, f"cannot import netCDF4/numpy to validate: {exc}"
    try:
        ds = Dataset(path)
    except Exception as exc:  # noqa: BLE001 - an HTML error page is not a NetCDF
        return False, f"not a readable NetCDF (server error page?): {exc}"
    try:
        for v in ("lat", "lon", "fw"):
            if v not in ds.variables:
                return False, f"missing variable {v!r} (got {list(ds.variables)})"
        fw = np.ma.filled(ds.variables["fw"][:], np.nan).astype("float64")
        lat = np.asarray(ds.variables["lat"][:], "float64")
        lon = np.asarray(ds.variables["lon"][:], "float64")
    finally:
        ds.close()
    # load_swamps_fw reads 1-D lat/lon and a 2-D fw of shape (lat, lon), so reject
    # anything else here rather than downstream. A leftover T axis (fw is 3-D)
    # means the monthly average did not collapse and the URL expression is wrong.
    if lat.ndim != 1 or lon.ndim != 1:
        return False, f"lat/lon must be 1-D (got {lat.ndim}-D and {lon.ndim}-D)"
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        return False, "lat/lon contain non-finite values"
    if fw.ndim != 2 or fw.shape != (lat.size, lon.size):
        return False, (f"fw must be 2-D ({lat.size} x {lon.size}); got shape "
                       f"{fw.shape} (a leftover time axis means T did not average)")
    finite = np.isfinite(fw)
    if not finite.any():
        return False, "fw is entirely missing"
    lo, hi = float(np.nanmin(fw)), float(np.nanmax(fw))
    if lo < -0.001 or hi > 1.001:
        return False, f"fw out of the fractional [0, 1] range: [{lo}, {hi}]"
    return True, (f"{lat.size}x{lon.size} grid, {100 * finite.mean():.1f}% valid, "
                  f"fw in [{lo:.3f}, {hi:.3f}]")


def fetch_month(ym, out_dir, force=False):
    if not re.fullmatch(r"\d{6}", ym) or not 1 <= int(ym[4:6]) <= 12:
        print(f"  {ym}: NOT A VALID YYYYMM, skipped")
        return "bad"
    ymi = int(ym)
    if ymi < FIRST_MONTH or ymi > LAST_MONTH:
        print(f"  {ym}: OUT OF RANGE (record is {FIRST_MONTH} to {LAST_MONTH}), skipped")
        return "range"
    path = os.path.join(out_dir, f"swamps_fw_{ym}.nc")
    if os.path.exists(path) and not force:
        ok, msg = validate(path)
        if ok:
            print(f"  {ym}: cached ({msg})")
            return "cached"
        print(f"  {ym}: cached file invalid ({msg}), refetching")
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(month_mean_url(ym), tmp)
    except Exception as exc:  # noqa: BLE001
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"  {ym}: download failed ({exc})")
        return "miss"
    ok, msg = validate(tmp)
    if not ok:
        os.remove(tmp)
        print(f"  {ym}: rejected ({msg})")
        return "miss"
    os.replace(tmp, path)
    print(f"  {ym}: wrote {os.path.basename(path)} ({msg})")
    return "got"


def main():
    args = sys.argv[1:]
    force = "--force" in args
    args = [a for a in args if a != "--force"]
    out_dir = "../data/swamps"
    if "--out" in args:
        i = args.index("--out")
        out_dir = args[i + 1]
        args = args[:i] + args[i + 2:]
    months = args
    if not months:
        print(__doc__)
        return 1
    os.makedirs(out_dir, exist_ok=True)
    socket_timeout(180)
    counts = {"got": 0, "cached": 0, "miss": 0, "range": 0, "bad": 0}
    for ym in months:
        counts[fetch_month(ym, out_dir, force=force)] += 1
    print(f"{out_dir}: {counts['got']} downloaded, {counts['cached']} cached, "
          f"{counts['miss']} failed, {counts['range']} out of range, "
          f"{counts['bad']} malformed")
    # Any requested month that did not yield a file is a non-zero exit: a caller
    # asking for months and getting none must not read that as success.
    return 1 if (counts["miss"] or counts["range"] or counts["bad"]) else 0


def socket_timeout(seconds):
    import socket
    socket.setdefaulttimeout(seconds)


if __name__ == "__main__":
    sys.exit(main())
