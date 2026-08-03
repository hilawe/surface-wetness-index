"""Reader and gridder for CSU SSM/I TDR BASE antenna-temperature swaths.

The decision tree was written for the operational Temperature Data Record, whose
values are antenna temperatures, while the modern gridded record this project
otherwise uses supplies brightness temperatures. Testing which convention the
recovered code expects needs authentic antenna temperatures on the same grid as
the brightness temperatures, which is what this module produces.

Input is the CSU TDR BASE product as distributed by NOAA CLASS: one NetCDF
granule per orbit, roughly fifteen per day, carrying

- five low-frequency channels (`ta19v ta19h ta22v ta37v ta37h`) on a
  (nscan, 64) swath with `lat_lores` and `lon_lores`, and
- the two 85 GHz channels (`ta85v ta85h`) on a (2*nscan, 128) swath with its
  own `lat_hires` and `lon_hires`.

Gridding choices, and why:

- The target is the CSU grid exactly, 720 by 1440 at 0.25 degree with latitude
  centers from -89.875 and longitude centers from 0.125 in a 0 to 360
  convention, so a gridded antenna temperature and the CSU brightness
  temperature for the same day are directly comparable cell for cell.
- Cells are filled with the mean of the pixels falling in them, matching the
  CSU product's own `cell_methods` of "area: mean". The legacy operational
  decoder instead assigned the last pixel to land in a cell. Mirroring that
  here would confound the gridding rule with the antenna-versus-brightness
  question this module exists to answer, so the CSU rule is used.
- The two swaths are binned independently, each with its own geolocation, as
  the legacy decoder did for its imager and environment scans. The 85 GHz
  channels are not first averaged into the low-frequency footprint.
- Ascending and descending are separated, as in the CSU product. The pass comes
  from the sign of the along-track gradient of a smoothed scan-center latitude,
  with any residual ambiguity at the orbital turning points resolved by carrying
  the previous scan's direction forward. Taking the sign of the raw gradient
  flips spuriously around the poles, where consecutive scan latitudes can differ
  by less than their own noise. The `spacecraft_lat` variable is documented as
  repeating and is not usable.
- Granules are assigned to a day by the UTC time of each scan, not by the date
  in the archive name. An archive named for one day contains an orbit that runs
  past midnight, so binning whole granules would mix two UTC days into one grid
  and would not line up with the CSU daily product, which covers 00:00 to 23:59
  UTC.
- Filtering scans by time is only half of the requirement. A granule is filed
  under the day on which it starts, so the scans that fall in the first minutes
  of a day live in the previous day's archive. Every daily archive inspected
  ends with such a granule, and the missing window runs from 1 to 100 minutes
  depending on orbit phase. `grid_day` therefore expects the caller to offer the
  neighbouring days' granules as well and lets the time filter select from the
  union. Feeding it a single archive silently produces a short day.
- Because granules from neighbouring days are routinely offered, a granule whose
  scan times cannot be read is skipped rather than binned whole. Binning it
  whole would import a foreign orbit into the day, which is a worse failure than
  losing one granule.
"""

import datetime as _dt

import glob
import os
from typing import NamedTuple

import numpy as np

from .channels import N_CHANNELS

NLAT, NLON = 720, 1440
DLAT = DLON = 0.25
LAT0, LON0 = -90.0, 0.0
FILL = -9999.9

# Basist order P1..P7 = 19V 19H 22V 37V 37H 85V 85H. The first five come from
# the low-resolution swath, the last two from the high-resolution swath.
LORES_VARS = ["ta19v", "ta19h", "ta22v", "ta37v", "ta37h"]
HIRES_VARS = ["ta85v", "ta85h"]
CHANNEL_NAMES = LORES_VARS + HIRES_VARS


def grid_centers():
    """The CSU grid's latitude and longitude cell centers."""
    lat = LAT0 + DLAT * (np.arange(NLAT) + 0.5)
    lon = LON0 + DLON * (np.arange(NLON) + 0.5)
    return lat, lon


def _cell_index(lat, lon):
    """Cell indices for pixel latitudes and longitudes, and a validity mask."""
    ok = np.isfinite(lat) & np.isfinite(lon)
    # Index off finite stand-ins so a non-finite geolocation does not raise an
    # invalid-cast warning. Those pixels are already excluded by `ok`.
    safe_lat = np.where(ok, lat, 0.0)
    safe_lon = np.where(ok, lon, 0.0)
    ilat = np.floor((safe_lat - LAT0) / DLAT).astype(np.int64)
    ilon = np.floor((np.mod(safe_lon, 360.0) - LON0) / DLON).astype(np.int64)
    # The grid's latitude interval is closed at both poles. Flooring maps the
    # exact north pole to row NLAT, past the last row, so fold that single
    # boundary value back rather than silently dropping it. Longitude needs no
    # counterpart, since np.mod already folds 360.0 onto 0.0. Only the exact
    # value moves; anything above 90.0 stays rejected by the range check below.
    ilat = np.where(safe_lat == 90.0, NLAT - 1, ilat)
    ok &= (ilat >= 0) & (ilat < NLAT) & (ilon >= 0) & (ilon < NLON)
    return ilat, ilon, ok


SMOOTH_SCANS = 9          # about 17 seconds of track, far shorter than an orbit


def _ascending(lat_center):
    """Per-scan ascending mask from a smoothed along-track latitude gradient.

    The raw gradient changes sign spuriously near the orbital turning points,
    where successive scan latitudes differ by less than their own noise, which
    would put scan lines in the wrong pass. Smoothing first removes most of
    that, and any remaining zero-gradient scans inherit the previous scan's
    direction rather than defaulting to descending.
    """
    n = lat_center.size
    if n < 2:
        return np.ones(n, bool)
    k = min(SMOOTH_SCANS, n if n % 2 else n - 1)
    if k >= 3:
        pad = k // 2
        padded = np.concatenate([np.repeat(lat_center[0], pad), lat_center,
                                 np.repeat(lat_center[-1], pad)])
        smooth = np.convolve(padded, np.ones(k) / k, mode="valid")
    else:
        smooth = lat_center
    g = np.gradient(smooth)
    sign = np.sign(g)
    # Carry the last unambiguous direction across flat stretches.
    idx = np.where(sign != 0, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    sign = sign[idx]
    if sign[0] == 0:                       # nothing before the first turn
        nz = np.flatnonzero(sign != 0)
        if nz.size:
            sign[:nz[0]] = sign[nz[0]]
    return sign > 0


def _day_start(day):
    """Seconds from the product epoch to 00:00 UTC on a YYYYMMDD date."""
    return (_dt.datetime.strptime(day, "%Y%m%d")
            - _dt.datetime(1987, 1, 1)).total_seconds()


def _utc_day_mask(ds, day, nscan):
    """Per-scan mask selecting scans whose UTC date is `day` (YYYYMMDD).

    Returns None when the granule carries no usable scan time, in which case
    the caller keeps every scan.
    """
    if day is None or "xtime" not in ds.variables:
        return None
    x = np.asarray(ds["xtime"][:], np.float64)
    if x.size != nscan:
        return None
    t0 = _day_start(day)
    return (x >= t0) & (x < t0 + 86400.0)


class _Accumulator:
    """Per-pass running sums and counts for the seven channels."""

    def __init__(self):
        self.total = np.zeros((2, NLAT, NLON, N_CHANNELS), np.float64)
        self.count = np.zeros((2, NLAT, NLON, N_CHANNELS), np.int64)

    def add(self, p, ch, ilat, ilon, values):
        flat = ilat * NLON + ilon
        # A negative index would not raise here, numpy indexes it silently
        # from the array's end, so the in-grid property has to be asserted
        # rather than assumed. Callers screen with the _cell_index mask, and
        # this is the backstop for the caller that forgets.
        assert flat.size == 0 or (flat.min() >= 0
                                  and flat.max() < NLAT * NLON), \
            "pixel index outside the grid reached the accumulator"
        np.add.at(self.total[p, :, :, ch].reshape(-1), flat, values)
        np.add.at(self.count[p, :, :, ch].reshape(-1), flat, 1)

    def means(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(self.count > 0, self.total / np.maximum(self.count, 1),
                         np.nan)
        return m, self.count


# Scans run about every 3.8 seconds and granules abut, so any real break in a
# day's coverage is far longer than this.
GAP_TOLERANCE = 60.0


class GranuleResult(NamedTuple):
    """What one granule gave a day."""

    binned: int                # pixel values accumulated
    kept: int                  # scans the time filter placed in this day
    times: object              # kept scan times, or None when unavailable


class DayReport(NamedTuple):
    """What a day was built from, what was refused, and what it covers.

    `complete` deliberately depends on measured temporal coverage rather than on
    the absence of refusals. A day can lose most of its data without any input
    being refused, for instance when the target archive is empty and only a
    neighbour's midnight tail reaches the day, so a flag that meant only "nothing
    was refused" told a consumer almost nothing.
    """

    used: list                 # granule paths that contributed at least a pixel
    rejected: list             # (path, reason) for every input refused
    empty: list                # (path, reason) for scans in the day with no data
    gaps: list                 # (start, end) seconds from midnight, uncovered
    covered: float             # seconds of the day carrying at least one scan

    @property
    def complete(self):
        """True when nothing was refused and the whole UTC day is covered."""
        return not self.rejected and not self.empty and not self.gaps

    @property
    def covered_fraction(self):
        return self.covered / 86400.0


def _coverage(times, day):
    """Seconds of `day` covered by these scan times, and the gaps left over."""
    if day is None or not len(times):
        return 0.0, []
    rel = np.sort(np.concatenate(times)) - _day_start(day)
    edges = np.concatenate([[0.0], rel, [86400.0]])
    span = np.diff(edges)
    gaps = [(float(edges[i]), float(edges[i + 1]))
            for i in np.flatnonzero(span > GAP_TOLERANCE)]
    return 86400.0 - float(sum(b - a for a, b in gaps)), gaps


def add_granule(acc, path, day=None, require_time=True, rejected=None):
    """Bin one orbit granule into the accumulator, returning pixels binned.

    `day` is a YYYYMMDD string. When given, only scans whose UTC time falls in
    that day are binned, so a granule spanning midnight contributes to each day
    only where it belongs, and a granule offered from a neighbouring day's
    archive contributes only its scans that reach into this day.

    With `day` given and `require_time` left true, a swath is refused unless its
    scan times can be read, its scan count lines up with those times, and its
    quality flags are present and correctly shaped. Refusing is the safe choice
    because neighbouring archives are routinely offered, so binning a swath that
    cannot be placed would import a foreign orbit. Callers that already know a
    granule belongs to the day can pass `require_time=False` to bin it whole.

    Every refusal is appended to `rejected` as a (path, reason) pair when a list
    is supplied, so the caller can report an incomplete day rather than emitting
    one that looks whole.
    """
    import netCDF4 as nc

    def refuse(reason):
        if rejected is not None:
            rejected.append((path, reason))

    ds = nc.Dataset(path)
    n_binned = 0
    kept, kept_times = 0, None
    try:
        nscan_lores = ds.dimensions["nscan_lores"].size
        day_lores = _utc_day_mask(ds, day, nscan_lores)
        if day is not None and day_lores is None:
            if require_time:
                refuse("no usable scan times")
                return GranuleResult(0, 0, None)
            refuse("no usable scan times, binned whole on request")
        if day_lores is not None:
            kept = int(day_lores.sum())
            if not kept:
                return GranuleResult(0, 0, None)
            kept_times = np.asarray(ds["xtime"][:], np.float64)[day_lores]
        else:
            kept = nscan_lores
        for group, varnames, latname, lonname, qname in (
                ("lores", LORES_VARS, "lat_lores", "lon_lores", "qflag_lores"),
                ("hires", HIRES_VARS, "lat_hires", "lon_hires", "qflag_hires")):
            lat = np.asarray(ds[latname][:], np.float64)
            lon = np.asarray(ds[lonname][:], np.float64)
            ilat, ilon, ok = _cell_index(lat, lon)
            asc = _ascending(lat[:, lat.shape[1] // 2])
            if day_lores is not None:
                # The high-resolution swath has two scans per low-resolution
                # scan and carries no time of its own, so it inherits the
                # low-resolution scan's UTC day. The pairing is checked rather
                # than assumed, because truncating the repeated mask to fit
                # would silently place scans in the wrong day.
                if group == "lores":
                    keep = (day_lores if day_lores.size == lat.shape[0]
                            else None)
                else:
                    keep = (np.repeat(day_lores, 2)
                            if lat.shape[0] == 2 * day_lores.size else None)
                if keep is not None:
                    ok &= keep[:, None]
                elif require_time:
                    refuse(f"{group} scans ({lat.shape[0]}) do not pair with "
                           f"the {day_lores.size} scan times")
                    continue

            # An absent or misshapen quality flag would otherwise pass flagged
            # values through unfiltered, which looks like good data.
            qflag = (np.asarray(ds[qname][:]) if qname in ds.variables
                     else None)
            usable_q = (qflag is not None and qflag.ndim == 2
                        and qflag.shape[0] == lat.shape[0]
                        and qflag.shape[1] >= len(varnames))
            if not usable_q:
                shape = "absent" if qflag is None else f"shape {qflag.shape}"
                if require_time:
                    refuse(f"{group} quality flags unusable ({shape})")
                    continue
                refuse(f"{group} quality flags unusable ({shape}), "
                       f"binned unfiltered on request")

            for k, name in enumerate(varnames):
                ch = CHANNEL_NAMES.index(name)
                ta = np.asarray(ds[name][:], np.float64)
                good = ok & np.isfinite(ta) & (ta > 50.0) & (ta < 400.0)
                if usable_q:
                    good &= (qflag[:, k] == 0)[:, None]
                for p, sel in ((0, asc), (1, ~asc)):
                    m = good & sel[:, None]
                    if not m.any():
                        continue
                    acc.add(p, ch, ilat[m], ilon[m], ta[m])
                    n_binned += int(m.sum())
    finally:
        ds.close()
    return GranuleResult(n_binned, kept, kept_times)


def grid_day(paths, day=None, require_time=True):
    """Grid one UTC day of orbit granules.

    `paths` should carry the target day's granules and those of the day before,
    because the granule holding the first minutes of a day is filed under the
    previous day. The scan-time filter selects from the union, so offering a
    neighbour costs only the read.

    The same file offered twice is read once. Granule identity across different
    archives is the caller's business, since only the caller knows whether two
    paths name the same orbit.

    Returns (lat, lon, means, counts, report), where `report` is a DayReport
    naming what contributed, what was refused, and what the day covers.

    A granule that cannot be read is refused rather than allowed to end the run,
    since one damaged file in an archive should not cost every other day.
    """
    acc = _Accumulator()
    used, rejected, empty, times, seen = [], [], [], [], set()
    for p in sorted(paths):
        canonical = os.path.realpath(p)
        if canonical in seen:
            continue
        seen.add(canonical)
        try:
            r = add_granule(acc, p, day=day, require_time=require_time,
                            rejected=rejected)
        except Exception as exc:                   # unreadable or malformed
            rejected.append((p, f"unreadable, {type(exc).__name__}: {exc}"))
            continue
        if r.binned:
            used.append(p)
            if r.times is not None:
                times.append(r.times)
        elif r.kept:
            # Scans of this granule fall in the day, yet none survived the
            # quality and range screening, which is a coverage loss that no
            # refusal would otherwise record.
            empty.append((p, f"{r.kept} scans in the day, no usable pixels"))
    lat, lon = grid_centers()
    means, counts = acc.means()
    covered, gaps = _coverage(times, day)
    return lat, lon, means, counts, DayReport(used, rejected, empty, gaps,
                                              covered)


def write_grid(out_path, lat, lon, means, counts, source_files, satellite,
               report=None):
    """Write a gridded antenna-temperature day in the CSU grid layout.

    The file is built beside its destination and moved into place once it is
    closed, so an interrupted rebuild leaves the previous day intact rather than
    a truncated replacement.

    `report` is the DayReport that built the day. Its refusals, its granules
    that yielded nothing, and its measured temporal coverage are all written
    into the file, so a consumer reading only the file can tell a whole day from
    a partial one. Console output is not durable provenance.
    """
    import netCDF4 as nc

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = f"{out_path}.partial"
    try:
        ds = nc.Dataset(tmp_path, "w", format="NETCDF4_CLASSIC")
        _fill_grid(ds, lat, lon, means, counts, source_files, satellite,
                   report)
        os.replace(tmp_path, out_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return out_path


def _fill_grid(ds, lat, lon, means, counts, source_files, satellite, report):
    """Write the variables and metadata into an open dataset, then close it."""
    try:
        ds.createDimension("lat", NLAT)
        ds.createDimension("lon", NLON)
        v = ds.createVariable("lat", "f8", ("lat",))
        v.units = "degrees_north"; v.standard_name = "latitude"; v[:] = lat
        v = ds.createVariable("lon", "f8", ("lon",))
        v.units = "degrees_east"; v.standard_name = "longitude"; v[:] = lon

        for ch, name in enumerate(CHANNEL_NAMES):
            for p, tag in ((0, "asc"), (1, "dsc")):
                var = ds.createVariable(f"{name}_{tag}", "f4", ("lat", "lon"),
                                        fill_value=FILL, zlib=True)
                var.units = "kelvin"
                var.long_name = (f"Gridded {name[2:]} antenna temperature, "
                                 f"{'ascending' if p == 0 else 'descending'}")
                var.cell_methods = "area: mean"
                var[:] = np.where(np.isfinite(means[p, :, :, ch]),
                                  means[p, :, :, ch], FILL)
                cvar = ds.createVariable(f"n_{name}_{tag}", "i4",
                                         ("lat", "lon"), zlib=True)
                cvar.long_name = f"Pixel count contributing to {name}_{tag}"
                cvar[:] = counts[p, :, :, ch]

        ds.title = "Gridded SSM/I TDR BASE antenna temperatures"
        ds.summary = (
            "Antenna temperatures from the CSU SSM/I TDR BASE swath product, "
            "binned to the CSU 0.25 degree grid as the mean of contributing "
            "pixels, ascending and descending separated. These are ANTENNA "
            "temperatures, not brightness temperatures, and exist to test which "
            "input convention the recovered decision tree expects.")
        ds.satellite = satellite
        ds.grid = "720 x 1440, 0.25 degree, lat from -89.875, lon from 0.125"
        ds.source_granules = " ".join(sorted(os.path.basename(p)
                                             for p in source_files))
        ds.complete = "true" if (report is None or report.complete) else "false"
        ds.rejected_granules = "" if report is None else "; ".join(
            f"{os.path.basename(p)}: {reason}" for p, reason in report.rejected)
        ds.granules_without_usable_data = "" if report is None else "; ".join(
            f"{os.path.basename(p)}: {reason}" for p, reason in report.empty)
        if report is not None:
            ds.coverage_seconds = float(report.covered)
            ds.coverage_fraction = float(report.covered_fraction)
            ds.coverage_gaps_utc = "; ".join(
                f"{_hhmmss(a)}-{_hhmmss(b)}" for a, b in report.gaps)
        ds.coverage_note = (
            "complete is true only when nothing was refused, every contributing "
            "granule yielded data, and scans span the whole UTC day")
    finally:
        ds.close()


def _hhmmss(seconds):
    """Seconds from midnight as HH:MM:SS, for the coverage-gap record."""
    s = int(round(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
