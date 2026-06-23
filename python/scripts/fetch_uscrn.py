"""Download USCRN daily01 station files for one or more years.

    python -m scripts.fetch_uscrn 2021 2022 2023 [--out DIR] [--workers W]

Downloads every station daily soil and meteorology file (CRND0103) for each year
from NOAA NCEI (open HTTPS, no authentication) into DIR (default ../data/uscrn),
W files at a time, with per-file retries. Cached files are kept. The U.S. Climate
Reference Network is NOAA's own research-grade reference network, so this is an
in-situ ground-truth source for the wetness validation.
"""

import concurrent.futures
import os
import re
import socket
import sys
import urllib.request

BASE = "https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01"


def station_files(year):
    idx = urllib.request.urlopen(f"{BASE}/{year}/").read().decode("utf-8", "replace")
    return sorted(set(re.findall(rf'CRND0103-{year}-[A-Z]{{2}}_[^"]+?\.txt', idx)))


def _fetch(year, name, out):
    path = os.path.join(out, name)
    if os.path.exists(path):
        return "cached"
    for _ in range(3):
        try:
            urllib.request.urlretrieve(f"{BASE}/{year}/{name}", path)
            return "got"
        except Exception:  # noqa: BLE001 - retry, then report
            if os.path.exists(path):
                os.remove(path)
    return "miss"


def main():
    args = sys.argv[1:]
    out, workers = "../data/uscrn", 8
    if "--out" in args:
        i = args.index("--out"); out = args[i + 1]; args = args[:i] + args[i + 2:]
    if "--workers" in args:
        i = args.index("--workers"); workers = int(args[i + 1]); args = args[:i] + args[i + 2:]
    years = [int(a) for a in args]
    if not years:
        print(__doc__)
        return 1
    os.makedirs(out, exist_ok=True)
    socket.setdefaulttimeout(25)

    got = cached = miss = 0
    for year in years:
        names = station_files(year)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for status in ex.map(lambda n: _fetch(year, n, out), names):
                got += status == "got"
                cached += status == "cached"
                miss += status == "miss"
        print(f"{year}: {len(names)} stations listed")
    print(f"{out}: {got} downloaded, {cached} cached, {miss} gaps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
