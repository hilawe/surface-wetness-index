"""Grid antenna and brightness temperature through one identical operator.

The question this exists to answer is which input convention the recovered
decision tree was written for, antenna temperature from the Temperature Data
Record (TDR) or brightness temperature from the Fundamental Climate Data Record
(FCDR). Answering it needs the two conventions on the same cells, and the
obvious route, comparing a self-gridded antenna field against the official CSU
brightness grid, does not give that. The official grid uses a footprint-aware
resampler with one common support across channels, while binning footprint
centres reaches roughly a quarter fewer cells at the low frequencies and gives
each channel its own support. The difference between the two arms would then
carry the gridding operator as well as the convention.

The two products make a cleaner experiment possible. CSU distributes the same
orbits in both forms, and the granules pair one to one by orbit revision number
with identical array shapes, so pixel (i, j) in a TDR granule and pixel (i, j)
in its FCDR counterpart are the same observation expressed two ways. Verified on
F-13, 1998-07-15: correlations of 0.99974 to 0.99999 across the seven channels,
with mean brightness minus antenna offsets of about 2 to 6 kelvin.

So this module reads a matched pair and grids both arms through exactly one of
everything:

- One geolocation, the FCDR's. The two products do not agree here, because the
  FCDR is geolocation corrected relative to TDR BASE, by a median 3.78 km and up
  to 11.4 km against a 28 km cell. Using each product's own coordinates would
  put the same observation in different cells and reintroduce a difference that
  has nothing to do with the convention.
- One scan time, the FCDR's, which differs from the TDR's by about 1.9 seconds,
  below the 3.8 second scan cadence but enough to move a scan across midnight.
- One quality mask, the intersection of both products' screening. A pixel enters
  only if the TDR calls its scan and channel good and the FCDR calls the pixel
  good. The two schemes differ in shape, per scan and channel against per pixel,
  so neither alone would screen both arms.
- One pass assignment, one cell index, one set of contributing pixels per cell.

What is left varying between the two output grids is the radiometric convention
and nothing else. Neither arm reproduces the official CSU grid, and that is the
intended trade: the comparison is internally controlled rather than matched to a
product built by a different operator. Say so plainly wherever the result is
reported.
"""

import os
import re

import numpy as np

from .channels import N_CHANNELS
from .io_tdr_swath import (DayReport, GranuleResult, _Accumulator, _ascending,
                           _cell_index, _coverage, _day_start, _hhmmss,
                           grid_centers)

# Basist order P1..P7, with the name each product gives the channel.
LORES = [("ta19v", "fcdr_tb19v"), ("ta19h", "fcdr_tb19h"),
         ("ta22v", "fcdr_tb22v"), ("ta37v", "fcdr_tb37v"),
         ("ta37h", "fcdr_tb37h")]
HIRES = [("ta85v", "fcdr_tb85v"), ("ta85h", "fcdr_tb85h")]

# The FCDR quality flag is per pixel: 0 good, 1 to 99 minor, 100 and above set
# to missing. Only wholly clean pixels are taken, on both arms alike.
FCDR_GOOD = 0

REV = re.compile(r"_R(\d+)\.nc$")


def revision(path):
    """Orbit revision number from a granule filename, or None."""
    m = REV.search(os.path.basename(path))
    return m.group(1) if m else None


def pair_paths(tdr_paths, fcdr_paths):
    """Match TDR and FCDR granules by orbit revision.

    Returns (pairs, unmatched) where pairs is a list of (revision, tdr, fcdr)
    sorted by revision, and unmatched lists the paths with no counterpart. An
    unmatched granule is reported rather than dropped quietly, because a missing
    counterpart is a coverage loss in one arm only, which would bias the
    comparison in exactly the way it exists to avoid.
    """
    by_rev = {}
    for p in tdr_paths:
        r = revision(p)
        if r:
            by_rev.setdefault(r, [None, None])[0] = p
    for p in fcdr_paths:
        r = revision(p)
        if r:
            by_rev.setdefault(r, [None, None])[1] = p
    pairs, unmatched = [], []
    for r in sorted(by_rev):
        tdr, fcdr = by_rev[r]
        if tdr and fcdr:
            pairs.append((r, tdr, fcdr))
        else:
            unmatched.append(tdr or fcdr)
    return pairs, unmatched


def _day_mask(times, day):
    """Per-scan mask for the UTC day, or None when no day is requested."""
    if day is None:
        return None
    t0 = _day_start(day)
    return (times >= t0) & (times < t0 + 86400.0)


def add_pair(acc_ta, acc_tb, tdr_path, fcdr_path, day=None, rejected=None):
    """Bin one matched granule pair into two accumulators, identically.

    Returns a GranuleResult counting the pixel values binned into each arm (they
    are equal by construction), the scans kept, and their times.
    """
    import netCDF4 as nc

    def refuse(reason):
        if rejected is not None:
            rejected.append((tdr_path, reason))

    t = nc.Dataset(tdr_path)
    f = nc.Dataset(fcdr_path)
    binned, kept, kept_times = 0, 0, None
    try:
        if "scan_time_lores" not in f.variables:
            refuse("FCDR granule carries no scan_time_lores")
            return GranuleResult(0, 0, None)
        times = np.asarray(f["scan_time_lores"][:], np.float64)
        nscan = times.size
        if t.dimensions["nscan_lores"].size != nscan:
            refuse(f"scan counts differ, TDR "
                   f"{t.dimensions['nscan_lores'].size} against FCDR {nscan}")
            return GranuleResult(0, 0, None)

        day_lores = _day_mask(times, day)
        if day_lores is not None:
            kept = int(day_lores.sum())
            if not kept:
                return GranuleResult(0, 0, None)
            kept_times = times[day_lores]
        else:
            kept = nscan

        for group, names, latname, lonname, qname in (
                ("lores", LORES, "lat_lores", "lon_lores", "quality_lores"),
                ("hires", HIRES, "lat_hires", "lon_hires", "quality_hires")):
            # One geolocation for both arms, the corrected FCDR one.
            lat = np.asarray(f[latname][:], np.float64)
            lon = np.asarray(f[lonname][:], np.float64)
            if np.asarray(t[latname][:]).shape != lat.shape:
                refuse(f"{group} swath shapes differ between the products")
                continue
            ilat, ilon, ok = _cell_index(lat, lon)
            asc = _ascending(lat[:, lat.shape[1] // 2])

            if day_lores is not None:
                keep = (day_lores if group == "lores"
                        else np.repeat(day_lores, 2))
                if keep.size != lat.shape[0]:
                    refuse(f"{group} scans do not pair with the scan times")
                    continue
                ok &= keep[:, None]

            # One quality mask, the intersection of the two schemes.
            if qname not in f.variables:
                refuse(f"FCDR granule carries no {qname}")
                continue
            qvar = f[qname]
            # The flag carries a valid_range that will not cast to its own type,
            # so read it raw rather than let the library warn on every granule.
            qvar.set_auto_maskandscale(False)
            ok &= np.asarray(qvar[:]) == FCDR_GOOD
            tq = "qflag_lores" if group == "lores" else "qflag_hires"
            tflag = np.asarray(t[tq][:]) if tq in t.variables else None
            if tflag is None or tflag.ndim != 2 or \
                    tflag.shape[0] != lat.shape[0] or \
                    tflag.shape[1] < len(names):
                refuse(f"TDR {tq} unusable, so the two arms cannot be screened "
                       f"alike")
                continue

            for k, (ta_name, tb_name) in enumerate(names):
                ch = _channel_index(group, k)
                ta = np.asarray(t[ta_name][:], np.float64)
                tb = np.asarray(f[tb_name][:], np.float64)
                if ta.shape != lat.shape or tb.shape != lat.shape:
                    refuse(f"{ta_name} and {tb_name} shapes disagree")
                    continue
                # A pixel enters only where BOTH conventions are physical, so
                # the two grids rest on identical contributing pixels.
                good = (ok & (tflag[:, k] == 0)[:, None]
                        & np.isfinite(ta) & (ta > 50.0) & (ta < 400.0)
                        & np.isfinite(tb) & (tb > 50.0) & (tb < 400.0))
                for p, sel in ((0, asc), (1, ~asc)):
                    m = good & sel[:, None]
                    if not m.any():
                        continue
                    acc_ta.add(p, ch, ilat[m], ilon[m], ta[m])
                    acc_tb.add(p, ch, ilat[m], ilon[m], tb[m])
                    binned += int(m.sum())
    finally:
        t.close()
        f.close()
    return GranuleResult(binned, kept, kept_times)


def _channel_index(group, k):
    return k if group == "lores" else len(LORES) + k


def grid_day_paired(pairs, day=None):
    """Grid one UTC day of matched granule pairs.

    Returns (lat, lon, means_ta, means_tb, counts, report). The counts apply to
    both arms, since every cell rests on the same contributing pixels.
    """
    acc_ta, acc_tb = _Accumulator(), _Accumulator()
    used, rejected, empty, times, seen = [], [], [], [], set()
    for rev, tdr_path, fcdr_path in sorted(pairs):
        if rev in seen:
            continue
        seen.add(rev)
        try:
            r = add_pair(acc_ta, acc_tb, tdr_path, fcdr_path, day=day,
                         rejected=rejected)
        except Exception as exc:
            rejected.append((tdr_path,
                             f"unreadable, {type(exc).__name__}: {exc}"))
            continue
        if r.binned:
            used.append(tdr_path)
            if r.times is not None:
                times.append(r.times)
        elif r.kept:
            empty.append((tdr_path,
                          f"{r.kept} scans in the day, no usable pixels"))
    lat, lon = grid_centers()
    means_ta, counts = acc_ta.means()
    means_tb, counts_tb = acc_tb.means()
    # The two arms are screened together, so any divergence here is a defect.
    assert np.array_equal(counts, counts_tb), \
        "the two arms binned different pixels, which the design forbids"
    covered, gaps = _coverage(times, day)
    return (lat, lon, means_ta, means_tb, counts,
            DayReport(used, rejected, empty, gaps, covered))


def write_paired_grid(out_path, lat, lon, means_ta, means_tb, counts,
                      source_files, satellite, report=None):
    """Write both arms of a paired day into one file.

    Both conventions live in one file on purpose. They rest on identical
    contributing pixels, and keeping them together makes that inseparable, so no
    later step can pair a day of one arm with a different day of the other.
    """
    import netCDF4 as nc

    from .io_tdr_swath import FILL, NLAT, NLON, _hhmmss

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = f"{out_path}.partial"
    try:
        ds = nc.Dataset(tmp_path, "w", format="NETCDF4_CLASSIC")
        try:
            ds.createDimension("lat", NLAT)
            ds.createDimension("lon", NLON)
            v = ds.createVariable("lat", "f8", ("lat",))
            v.units = "degrees_north"
            v.standard_name = "latitude"
            v[:] = lat
            v = ds.createVariable("lon", "f8", ("lon",))
            v.units = "degrees_east"
            v.standard_name = "longitude"
            v[:] = lon

            for ch in range(N_CHANNELS):
                band = (LORES + HIRES)[ch][0][2:]      # "19v", "85h", and so on
                for p, tag in ((0, "asc"), (1, "dsc")):
                    for arm, field in (("ta", means_ta), ("tb", means_tb)):
                        var = ds.createVariable(f"{arm}{band}_{tag}", "f4",
                                                ("lat", "lon"),
                                                fill_value=FILL, zlib=True)
                        var.units = "kelvin"
                        kind = ("antenna" if arm == "ta" else "brightness")
                        var.long_name = (
                            f"Gridded {band} {kind} temperature, "
                            f"{'ascending' if p == 0 else 'descending'}")
                        var.cell_methods = "area: mean"
                        m = field[p, :, :, ch]
                        var[:] = np.where(np.isfinite(m), m, FILL)
                    cvar = ds.createVariable(f"n_{band}_{tag}", "i4",
                                             ("lat", "lon"), zlib=True)
                    cvar.long_name = (f"Pixel count contributing to {band} "
                                      f"{tag}, identical for both arms")
                    cvar[:] = counts[p, :, :, ch]

            ds.title = "Paired SSM/I antenna and brightness temperature grid"
            ds.summary = (
                "Antenna temperatures from the CSU TDR BASE swath product and "
                "brightness temperatures from the matching CSU FCDR swath "
                "product, gridded through one identical operator: the FCDR "
                "geolocation, the FCDR scan times, the intersection of both "
                "quality screens, one pass assignment and one set of "
                "contributing pixels per cell. Only the radiometric convention "
                "differs between the two arms. Neither arm reproduces the "
                "official CSU daily grid, which uses a footprint-aware "
                "resampler; this file is an internally controlled contrast, "
                "not a match to that product.")
            ds.satellite = satellite
            ds.grid = ("720 x 1440, 0.25 degree, lat from -89.875, "
                       "lon from 0.125")
            ds.geolocation = "FCDR, geolocation corrected, used for both arms"
            ds.source_granules = " ".join(sorted(os.path.basename(p)
                                                 for p in source_files))
            ds.complete = ("true" if (report is None or report.complete)
                           else "false")
            ds.rejected_granules = "" if report is None else "; ".join(
                f"{os.path.basename(p)}: {r}" for p, r in report.rejected)
            ds.granules_without_usable_data = "" if report is None else "; ".join(
                f"{os.path.basename(p)}: {r}" for p, r in report.empty)
            if report is not None:
                ds.coverage_seconds = float(report.covered)
                ds.coverage_fraction = float(report.covered_fraction)
                ds.coverage_gaps_utc = "; ".join(
                    f"{_hhmmss(a)}-{_hhmmss(b)}" for a, b in report.gaps)
        finally:
            ds.close()
        os.replace(tmp_path, out_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return out_path
