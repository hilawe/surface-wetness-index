"""Load ESA CCI Soil Moisture (COMBINED) and composite to a monthly mean.

ESA CCI Soil Moisture is the community microwave soil-moisture climate data
record (merged active and passive retrievals, Dorigo et al.). It is delivered as
daily 0.25 degree global files on the CEDA archive, openly accessible over the
THREDDS HTTPServer with no API key or registration. Daily coverage is sparse
(only converged retrievals over unmasked land), so we composite a month of daily
files to a per-cell mean over flag-clean cells, then return it on our grid
orientation (latitude ascending, longitude 0..360) to match the WET product.

Masking uses flag == 0 (no quality flag set). The CCI flag is a bit field whose
first two bits are snow_coverage_or_temperature_below_zero and dense_vegetation,
so flag == 0 already drops frozen and densely vegetated cells. Those are exactly
the Basist index's known blind spots, so the comparison is made only over cells
where both fields are physically meaningful.

Unlike ERA5-Land (a reanalysis model), ESA CCI is observation-based and shares
the passive-microwave physics family with the wetness index, so it is the
strongest independent reference for a microwave surface-wetness detector.
"""

import glob
import os

import numpy as np

CEDA_FILESERVER = "https://data.cci.ceda.ac.uk/thredds/fileServer"
PRODUCT = "COMBINED"
VERSION = "v09.2"


def cci_filename(date, product=PRODUCT, version=VERSION):
    """Canonical ESA CCI SM daily filename for 'YYYYMMDD'."""
    return (f"ESACCI-SOILMOISTURE-L3S-SSMV-{product}-"
            f"{date}000000-f{version}.nc")


def cci_url(date, product=PRODUCT, version=VERSION):
    """CEDA THREDDS HTTPServer URL for a daily ESA CCI SM file ('YYYYMMDD')."""
    return (f"{CEDA_FILESERVER}/esacci/soil_moisture/data/daily_files/"
            f"{product}/{version}/{date[:4]}/{cci_filename(date, product, version)}")


def download_cci_month(month, data_dir, product=PRODUCT, version=VERSION,
                       timeout=180):
    """Download all daily ESA CCI SM files for 'YYYYMM' into data_dir.

    Open access (no auth). Cached files are kept; missing server days are skipped
    with a note. Returns the sorted list of local file paths present afterward.
    """
    import calendar
    import socket
    import urllib.request

    socket.setdefaulttimeout(timeout)
    os.makedirs(data_dir, exist_ok=True)
    year, mon = int(month[:4]), int(month[4:6])
    ndays = calendar.monthrange(year, mon)[1]
    for d in range(1, ndays + 1):
        date = f"{year:04d}{mon:02d}{d:02d}"
        path = os.path.join(data_dir, cci_filename(date, product, version))
        if os.path.exists(path):
            continue
        try:
            urllib.request.urlretrieve(cci_url(date, product, version), path)
        except Exception as exc:  # noqa: BLE001 - report and continue past gaps
            if os.path.exists(path):
                os.remove(path)
            print(f"  CCI {date}: not retrieved ({exc})")
    return _month_files(month, data_dir, product, version)


def _month_files(month, data_dir, product=PRODUCT, version=VERSION):
    pat = os.path.join(
        data_dir,
        f"ESACCI-SOILMOISTURE-L3S-SSMV-{product}-{month}*-f{version}.nc")
    return sorted(glob.glob(pat))


def _read_day(path):
    """Return (lat, lon, sm) for one CCI day in native orientation.

    Soil moisture is NaN wherever flag != 0 (any quality flag set), so only
    clean retrievals over snow-free, non-densely-vegetated land survive.
    """
    import netCDF4 as nc

    ds = nc.Dataset(path)
    try:
        lat = np.asarray(ds["lat"][:], np.float64)
        lon = np.asarray(ds["lon"][:], np.float64)
        sm = np.squeeze(np.ma.filled(ds["sm"][:], np.nan)).astype(np.float64)
        flag = np.squeeze(np.ma.filled(ds["flag"][:], -1)).astype(np.int64)
    finally:
        ds.close()
    return lat, lon, np.where(flag == 0, sm, np.nan)


def load_cci_monthly(month, data_dir, product=PRODUCT, version=VERSION,
                     download=True):
    """Composite ESA CCI SM to a monthly mean on our grid orientation.

    Returns (lat_asc, lon_0360, sm_mean, n_days):
      - sm_mean: per-cell mean of flag-clean daily soil moisture over the month
        (NaN where no valid day),
      - n_days: per-cell count of valid days that went into the mean.
    Latitude is ascending (south first) and longitude is 0..360 ascending, to
    match the WET product grid.
    """
    if download:
        paths = download_cci_month(month, data_dir, product, version)
    else:
        paths = _month_files(month, data_dir, product, version)
    if not paths:
        raise FileNotFoundError(
            f"no ESA CCI {product} {version} files for {month} in {data_dir}")

    lat = lon = ssum = ncount = None
    for path in paths:
        la, lo, sm = _read_day(path)
        if lat is None:
            lat, lon = la, lo
            ssum = np.zeros(sm.shape, np.float64)
            ncount = np.zeros(sm.shape, np.float64)
        valid = np.isfinite(sm)
        ssum[valid] += sm[valid]
        ncount[valid] += 1.0

    with np.errstate(invalid="ignore"):
        mean = np.where(ncount > 0, ssum / np.maximum(ncount, 1.0), np.nan)

    if lat[0] > lat[-1]:                         # north-first -> south-first
        lat = lat[::-1]; mean = mean[::-1, :]; ncount = ncount[::-1, :]
    lon = np.where(lon < 0, lon + 360.0, lon)    # -180..180 -> 0..360
    order = np.argsort(lon)
    return lat, lon[order], mean[:, order], ncount[:, order]
