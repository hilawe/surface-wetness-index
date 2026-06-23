"""Load USCRN daily soil moisture (5 cm) and aggregate to monthly station means.

The U.S. Climate Reference Network (USCRN) is NOAA's reference network of about
160 research-grade stations, mostly across the contiguous United States. The daily
product (daily01) reports volumetric soil moisture at 5, 10, 20, 50, and 100 cm. We
use the 5 cm layer, the surface layer the microwave wetness senses, average it to a
monthly station mean over valid days, and the validation pairs it with the wetness
index at the grid cell that contains each station. This is a point-to-pixel
comparison, with the representativeness caveat that a single station samples a
0.25 degree cell.

USCRN soil moisture is volumetric, like ERA5-Land and ESA CCI, so the expected
result is the same detector behavior: a wet versus dry detection contrast, with
weak agreement on magnitude. The value of USCRN is that it is in-situ ground truth,
the most rigorous reference type, from NOAA's own network.
"""

import glob
import os

# Zero-based column indices in the daily01 fixed field set (28 columns).
WBAN_COL = 0
DATE_COL = 1        # LST_DATE, YYYYMMDD
LON_COL = 3
LAT_COL = 4
SM5_COL = 18        # SOIL_MOISTURE_5_DAILY, volumetric m3 m-3
FILL = -9999.0


def load_station_monthly(data_dir, min_days=10):
    """Aggregate USCRN 5 cm soil moisture to monthly station means.

    Parses every CRND0103 daily file in data_dir, keeps valid 5 cm volumetric soil
    moisture (between 0 and 1), and averages it per station and calendar month.
    Returns a dict mapping the station WBAN number to a record with its longitude,
    latitude, and a months dict of YYYYMM to the monthly mean. Months with fewer
    than min_days valid days are dropped.
    """
    acc = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "CRND0103-*.txt"))):
        with open(path) as fh:
            for line in fh:
                f = line.split()
                if len(f) <= SM5_COL:
                    continue
                try:
                    lon = float(f[LON_COL]); lat = float(f[LAT_COL])
                    sm = float(f[SM5_COL])
                except ValueError:
                    continue
                if sm == FILL or not (0.0 <= sm <= 1.0):
                    continue
                wban = f[WBAN_COL]
                ym = f[DATE_COL][:6]
                st = acc.setdefault(wban, {"lon": lon, "lat": lat, "sum": {}, "n": {}})
                st["lon"], st["lat"] = lon, lat
                st["sum"][ym] = st["sum"].get(ym, 0.0) + sm
                st["n"][ym] = st["n"].get(ym, 0) + 1

    out = {}
    for wban, st in acc.items():
        months = {ym: st["sum"][ym] / st["n"][ym]
                  for ym in st["sum"] if st["n"][ym] >= min_days}
        if months:
            out[wban] = {"lon": st["lon"], "lat": st["lat"], "months": months}
    return out
