"""Download F-16 CSU FCDR-GRID daily brightness-temperature files for a year.

    python -m scripts.fetch_csu_f16 YEAR [--step N] [--workers W] [--out DIR]

Downloads every Nth day of each month (default 4, matching the existing 2023
set: days 1, 5, 9, ...) of CSU SSMIS FCDR-GRID F-16 files from NCEI (open HTTPS,
no authentication) into DIR (default ../data/f16_<YEAR>/), W files at a time.
Cached files are kept; server gaps are reported, not fatal.

F-16 is the operational current-data satellite for the revival and is present in
FCDR-GRID for all years except 2024 (which holds only F-18). FCDR-GRID is the
reprocessed, citable tree (1987 to 2025); for 2025 or later, use ICDR-GRID.
"""

import calendar
import concurrent.futures
import os
import socket
import sys
import urllib.request

BASE = ("https://www.ncei.noaa.gov/data/ssmis-brightness-temperature-csu/access/"
        "FCDR-GRID")


def fname(date):
    return f"CSU_SSMIS_FCDR-GRID_V02R00_F16_D{date}.nc"


def url(date):
    return f"{BASE}/{date[:4]}/{fname(date)}"


def _fetch_one(date, out, attempts=3):
    path = os.path.join(out, fname(date))
    if os.path.exists(path):
        return ("cached", date)
    last = ""
    for _ in range(attempts):
        try:
            urllib.request.urlretrieve(url(date), path)
            return ("got", date)
        except Exception as exc:  # noqa: BLE001 - retry, then report the gap
            last = str(exc)
            if os.path.exists(path):
                os.remove(path)
    return ("miss", f"{date} ({last})")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    year = int(sys.argv[1])
    step, workers = 4, 8
    out = f"../data/f16_{year}"
    rest = sys.argv[2:]
    if "--step" in rest:
        step = int(rest[rest.index("--step") + 1])
    if "--workers" in rest:
        workers = int(rest[rest.index("--workers") + 1])
    if "--out" in rest:
        out = rest[rest.index("--out") + 1]
    os.makedirs(out, exist_ok=True)
    socket.setdefaulttimeout(45)

    dates = []
    for mon in range(1, 13):
        ndays = calendar.monthrange(year, mon)[1]
        for day in range(1, ndays + 1, step):
            dates.append(f"{year:04d}{mon:02d}{day:02d}")

    got = cached = miss = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for status, info in ex.map(lambda d: _fetch_one(d, out), dates):
            if status == "got":
                got += 1
            elif status == "cached":
                cached += 1
            else:
                miss += 1
                print(f"  not retrieved: {info}")
    print(f"{out}: {got} downloaded, {cached} cached, {miss} gaps "
          f"({len(dates)} target days)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
