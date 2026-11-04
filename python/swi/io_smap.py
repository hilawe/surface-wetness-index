"""Load SMAP L-band soil moisture (SPL3SMP_E) and composite to a monthly mean.

SMAP (Soil Moisture Active Passive) retrieves soil moisture at L-band, 1.4 GHz.
That frequency shares no channel, no sensor, and no seasonal artifact family with
the 19 to 85 GHz imager channels the Basist wetness index uses, so SMAP is a
PHYSICALLY INDEPENDENT reference in a way ESA CCI COMBINED is not. The CCI
COMBINED product merges passive retrievals from many of the same DMSP channels
as the index. SMAP removes that shared path entirely.

A prediction that agreement with SMAP would be LOWER than agreement with ESA CCI
was recorded in docs/SMAP_Comparison_Scope.md before any number was produced. It
was NOT confirmed. State that outcome and nothing more: the measured ordering
does not identify the direction or magnitude of any shared-channel bias, and it
does not establish that reference quality caused the difference. A shared
brightness-temperature artifact could raise the index and the CCI retrieval
together, which would CREATE a contrast rather than blur one, and no result here
rules that out.

Product and access
------------------
Collection SPL3SMP_E version 006 (Enhanced L3 Radiometer Global Daily Soil
Moisture, 9 km EASE-Grid 2.0), NASA NSIDC DAAC, CMR concept id
C2938664763-NSIDC_CPRD. A full granule is about 629 MB, but only the AM soil
moisture field and its retrieval quality flag are needed. The collection exposes
an OPeNDAP endpoint that serves a variable-constrained subset at about 3.75 MB
per day, so a month costs roughly 116 MB rather than 19.5 GB.

The working request form is fragile and was established by probing. The response
suffix must be `.dap.nc4`, which returns compressed netCDF4. Both `.nc4` and
`.nc` return HTTP 400 from this server. Access needs a NASA Earthdata login,
read from ~/.netrc, plus a cookie jar to survive the authentication redirect,
which is why this module uses requests rather than the urllib calls the
unauthenticated fetchers use.

Overpass and grid
-----------------
The AM (descending, about 6 am local) retrieval is used, matched to the wetness
index descending pass. Both are early morning and near thermal equilibrium, so
the pairing is physically fair rather than merely convenient.

The 9 km EASE-Grid 2.0 global cylindrical grid is separable, meaning every row
carries one latitude and every column one longitude, verified against the
delivered coordinate arrays. Going from 9 km to the 0.25 degree product grid is
a COARSENING, so cells are AREA-BINNED (every fine cell whose center falls in a
target cell is averaged) rather than sampled nearest-neighbor, which would
discard most of the retrievals and add noise.
"""

import calendar
import os

import numpy as np

CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
OPENDAP = "https://opendap.earthdata.nasa.gov/collections"
CONCEPT_ID = "C2938664763-NSIDC_CPRD"
SHORT_NAME = "SPL3SMP_E"
VERSION = "006"
GROUP_AM = "Soil_Moisture_Retrieval_Data_AM"
FILL_MAX = -9000.0          # SMAP fill is -9999; anything below this is not data
COORDS_FILE = "smap_coords_9km.nc4"
GRID_SHAPE = (1624, 3856)   # SPL3SMP_E 9 km EASE-Grid 2.0 global


def _session():
    """A requests session that carries Earthdata auth and its redirect cookie.

    Credentials come from ~/.netrc, which requests reads on its own. The cookie
    jar on the session is what survives the URS authentication redirect; without
    it every granule request restarts the handshake.
    """
    import requests

    s = requests.Session()
    s.headers.update({"User-Agent": "surface-wetness-index/smap-validation"})
    return s


def granules_for_month(month, session=None):
    """Return [(YYYYMMDD, granule_ur)] for every SPL3SMP_E granule in 'YYYYMM'.

    Queried from the CMR metadata API, which is public and needs no auth, so the
    granule list is derived from the archive rather than assumed from a date
    range. Days the archive does not hold simply do not appear.
    """
    import re

    s = session or _session()
    year, mon = int(month[:4]), int(month[4:6])
    last = calendar.monthrange(year, mon)[1]
    params = {
        "short_name": SHORT_NAME,
        "version": VERSION,
        "temporal": f"{year:04d}-{mon:02d}-01T00:00:00Z,"
                    f"{year:04d}-{mon:02d}-{last:02d}T23:59:59Z",
        "page_size": 200,
    }
    r = s.get(CMR, params=params, timeout=120)
    r.raise_for_status()
    out = {}
    for item in r.json().get("items", []):
        ur = item["umm"].get("GranuleUR", "")
        m = re.search(r"_(\d{8})_", ur)
        if m and m.group(1).startswith(month):
            out[m.group(1)] = ur
    return sorted(out.items())


def _opendap_subset(session, granule_ur, variables, out_path, timeout=600):
    """Download an OPeNDAP variable subset to out_path. Returns bytes written.

    Raises on a non-200 response or on a body that is not netCDF4/HDF5, so a
    Hyrax error page delivered with a 200 cannot masquerade as data.
    """
    ce = "%3B".join(f"/{GROUP_AM}/{v}" for v in variables)
    url = f"{OPENDAP}/{CONCEPT_ID}/granules/{granule_ur}.dap.nc4?dap4.ce={ce}"
    tmp = out_path + ".part"
    with session.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
    with open(tmp, "rb") as fh:
        magic = fh.read(8)
    if magic != b"\x89HDF\r\n\x1a\n":
        os.remove(tmp)
        raise ValueError(
            f"{granule_ur}: response was not netCDF4/HDF5 (got {magic!r}). "
            "The server may have returned an error page with a 200 status.")
    # The signature alone is not enough. A response truncated after those eight
    # bytes still carries them, so open the file and require the variables that
    # were actually asked for before promoting it into the cache.
    try:
        verify_subset(tmp, variables)
    except Exception:
        os.remove(tmp)
        raise
    os.replace(tmp, out_path)
    return os.path.getsize(out_path)


def verify_subset(path, variables):
    """Open a downloaded subset and confirm it holds what was requested.

    Raises unless the file opens as HDF5, carries the AM group, and carries every
    requested variable as a 2-D array on the expected 9 km grid. Existence and a
    magic number are not evidence that a transfer completed.
    """
    import h5py

    with h5py.File(path, "r") as f:
        if GROUP_AM not in f:
            raise ValueError(f"{os.path.basename(path)}: missing group {GROUP_AM}")
        g = f[GROUP_AM]
        # Presence for every requested variable first, then shapes. A truncated
        # file is usually missing a whole variable, and reporting a shape
        # mismatch on the one that survived would name the wrong problem.
        missing = [v for v in variables if v not in g]
        if missing:
            raise ValueError(
                f"{os.path.basename(path)}: missing variable {', '.join(missing)}")
        for v in variables:
            shape = g[v].shape
            if len(shape) != 2 or shape != GRID_SHAPE:
                raise ValueError(
                    f"{os.path.basename(path)}: {v} has shape {shape}, "
                    f"expected {GRID_SHAPE}")
    return True


def fetch_coords(data_dir, session=None):
    """Fetch the 9 km grid coordinates once. They are static across granules."""
    path = os.path.join(data_dir, COORDS_FILE)
    if os.path.exists(path):
        return path
    s = session or _session()
    gl = granules_for_month("202307", s)
    if not gl:
        raise RuntimeError("no SPL3SMP_E granule available to read coordinates from")
    os.makedirs(data_dir, exist_ok=True)
    _opendap_subset(s, gl[0][1], ["latitude", "longitude"], path)
    return path


def load_coords(data_dir):
    """Return (lat1d_desc, lon1d_asc) for the 9 km EASE-Grid 2.0 global grid.

    The delivered arrays are 2D. This collapses them to 1D and ASSERTS the grid
    really is separable rather than assuming it, because a non-separable grid
    would make the binning below silently wrong.
    """
    import h5py

    with h5py.File(os.path.join(data_dir, COORDS_FILE), "r") as f:
        g = f[GROUP_AM]
        lat2 = np.asarray(g["latitude"][:], np.float64)
        lon2 = np.asarray(g["longitude"][:], np.float64)
    lat2 = np.where(lat2 < FILL_MAX, np.nan, lat2)
    lon2 = np.where(lon2 < FILL_MAX, np.nan, lon2)
    if not np.all(np.nanstd(lat2, axis=1) < 1e-6):
        raise ValueError("SMAP latitude is not constant along rows; grid is not separable")
    if not np.all(np.nanstd(lon2, axis=0) < 1e-6):
        raise ValueError("SMAP longitude is not constant along columns; grid is not separable")
    return np.nanmean(lat2, axis=1), np.nanmean(lon2, axis=0)


def fetch_month(month, data_dir, session=None, verbose=True):
    """Fetch the AM soil moisture and quality flag subsets for every day.

    Cached files are kept. Returns [(date, path)] for the days present locally.
    """
    s = session or _session()
    os.makedirs(data_dir, exist_ok=True)
    fetch_coords(data_dir, s)
    got = []
    for date, ur in granules_for_month(month, s):
        path = os.path.join(data_dir, f"smap_am_{date}.nc4")
        if not os.path.exists(path):
            try:
                n = _opendap_subset(s, ur, ["soil_moisture", "retrieval_qual_flag"], path)
                if verbose:
                    print(f"  SMAP {date}: {n / 1048576:.2f} MB")
            except Exception as exc:  # noqa: BLE001 - report the gap and continue
                if verbose:
                    print(f"  SMAP {date}: not retrieved ({exc})")
                continue
        got.append((date, path))
    return got


"""Quality-flag bits in retrieval_qual_flag, per the SPL3SMP_E v006 user guide.

Bit 0 is set when the retrieval is NOT recommended, bit 1 when it was NOT
attempted, and bit 2 when it was NOT successful. Zero means the good case in
every one of them.
"""
QF_NOT_RECOMMENDED = 0b001
QF_NOT_ATTEMPTED = 0b010
QF_NOT_SUCCESSFUL = 0b100

SCREENS = ("recommended", "successful", "none")


def _read_day(path, screen="recommended"):
    """Return the AM soil moisture field for one day, NaN where unusable.

    Three screens, because the sensitivity question and the headline question
    want different ones.

    - "recommended" keeps only bit 0 clear. This is the headline screen and it
      discards roughly 45 percent of retrievals, so callers report how much
      survives rather than treating it as free.
    - "successful" admits recommended AND uncertain retrievals while still
      requiring one that was attempted and succeeded. This is the defensible
      relaxed screen for a sensitivity run.
    - "none" applies no quality condition at all. It admits retrievals the
      product itself marks unsuccessful, about 7.5 percent of July's numeric
      values, so it is NOT a valid scientific sensitivity and exists only to
      quantify what the screens remove.
    """
    import h5py

    if screen not in SCREENS:
        raise ValueError(f"unknown screen {screen!r}, expected one of {SCREENS}")
    with h5py.File(path, "r") as f:
        g = f[GROUP_AM]
        sm = np.asarray(g["soil_moisture"][:], np.float64)
        qf = np.asarray(g["retrieval_qual_flag"][:])
    sm = np.where(sm < FILL_MAX, np.nan, sm)
    if screen == "recommended":
        sm = np.where((qf & QF_NOT_RECOMMENDED) == 0, sm, np.nan)
    elif screen == "successful":
        sm = np.where((qf & (QF_NOT_ATTEMPTED | QF_NOT_SUCCESSFUL)) == 0, sm, np.nan)
    return sm


def _bin_to_target(field, slat, slon, tlat, tlon):
    """Area-bin a fine field onto the coarser target grid.

    Every fine cell whose center falls inside a target cell contributes to that
    cell's mean. Returns (sum, count) so callers can composite across days
    before dividing.
    """
    dlat = float(tlat[1] - tlat[0])
    dlon = float(tlon[1] - tlon[0])

    # The longitude index wraps with a modulo, which is only safe on a grid that
    # spans the whole globe. Latitude gets an explicit range test below and is
    # therefore safe on any span, but a modulo on a REGIONAL longitude grid would
    # silently fold far-away meridians into the domain instead of discarding
    # them, and the result would look ordinary. This project's product grid is
    # global, so the assumption holds, and asserting it is what stops the day
    # somebody reuses this loader on a subset.
    if not np.isclose(tlon.size * dlon, 360.0, atol=abs(dlon)):
        raise ValueError(
            f"target longitude grid spans {tlon.size * dlon:g} degrees, not 360. "
            "The wrap-around in this binning assumes a global grid.")

    ilat = np.rint((slat - tlat[0]) / dlat).astype(np.int64)
    # The modulo alone normalizes longitude, for negative values and for the seam
    # at 360 alike. An explicit "add 360 where negative" step used to sit here as
    # well, and a mutation run showed the two were redundant, each covering the
    # other so neither could be observed failing. One mechanism is kept.
    ilon = np.rint((slon - tlon[0]) / dlon).astype(np.int64) % tlon.size

    ok_lat = (ilat >= 0) & (ilat < tlat.size)
    I = np.broadcast_to(ilat[:, None], field.shape)
    J = np.broadcast_to(ilon[None, :], field.shape)
    valid = np.isfinite(field) & np.broadcast_to(ok_lat[:, None], field.shape)

    flat = (I[valid] * tlon.size + J[valid])
    n = tlat.size * tlon.size
    ssum = np.bincount(flat, weights=field[valid], minlength=n).reshape(tlat.size, tlon.size)
    scnt = np.bincount(flat, minlength=n).reshape(tlat.size, tlon.size).astype(np.float64)
    return ssum, scnt


def month_files(month, data_dir):
    """Return [(date, path)] for the cached days OF THIS MONTH only.

    Filtering by month is not cosmetic. Globbing every `smap_am_*` file would
    silently fold any other month sharing the directory into the composite while
    the result file still named this one.
    """
    out = []
    for name in sorted(os.listdir(data_dir)):
        if not (name.startswith("smap_am_") and name.endswith(".nc4")):
            continue
        date = name[len("smap_am_"):-len(".nc4")]
        if len(date) == 8 and date.isdigit() and date.startswith(month):
            out.append((date, os.path.join(data_dir, name)))
    return out


def load_smap_monthly(month, data_dir, target_lat, target_lon,
                      download=True, screen="recommended", verbose=True):
    """Composite SMAP AM soil moisture to a monthly mean on the target grid.

    Returns (sm_mean, n_days, stats):
      - sm_mean: per-cell mean over days that had any retrieval (NaN elsewhere),
      - n_days:  per-cell count of contributing days,
      - stats:   retention numbers, so the cost of the quality screen is
                 reported rather than hidden.

    Each day is screened and binned to the target grid FIRST, then the daily
    target fields are averaged, which matches how the ESA CCI comparison forms
    its monthly mean and keeps the two numbers comparable.
    """
    days = fetch_month(month, data_dir, verbose=verbose) if download \
        else month_files(month, data_dir)
    if not days:
        raise FileNotFoundError(f"no SMAP SPL3SMP_E files for {month} in {data_dir}")

    slat, slon = load_coords(data_dir)
    tlat = np.asarray(target_lat, np.float64)
    tlon = np.asarray(target_lon, np.float64)

    acc = np.zeros((tlat.size, tlon.size), np.float64)
    nday = np.zeros((tlat.size, tlon.size), np.float64)
    n_numeric = n_kept = n_successful = 0
    for _, path in days:
        n_numeric += int(np.sum(np.isfinite(_read_day(path, screen="none"))))
        n_successful += int(np.sum(np.isfinite(_read_day(path, screen="successful"))))
        sm = _read_day(path, screen=screen)
        n_kept += int(np.sum(np.isfinite(sm)))
        ssum, scnt = _bin_to_target(sm, slat, slon, tlat, tlon)
        with np.errstate(invalid="ignore", divide="ignore"):
            daily = np.where(scnt > 0, ssum / np.maximum(scnt, 1.0), np.nan)
        good = np.isfinite(daily)
        acc[good] += daily[good]
        nday[good] += 1.0

    with np.errstate(invalid="ignore"):
        mean = np.where(nday > 0, acc / np.maximum(nday, 1.0), np.nan)
    stats = {
        "screen": screen,
        "n_days_used": len(days),
        # numeric values carry retrievals the product marks unsuccessful, so this
        # is a denominator for describing the screens, not a usable population.
        "fine_numeric_values": n_numeric,
        "fine_successful_retrievals": n_successful,
        "fine_kept_by_screen": n_kept,
        "kept_fraction_of_successful": (n_kept / n_successful) if n_successful
        else float("nan"),
        "target_cells_with_data": int(np.sum(nday > 0)),
    }
    return mean, nday, stats
