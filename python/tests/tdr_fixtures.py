"""Synthetic CSU TDR BASE granules for the swath reader and gridder tests.

The real archives are large and not in the repository, so these build granules
with the same variable names, dimension names and shapes, letting the reader run
its actual code path. Every pixel in a scan shares that scan's geolocation, so a
scan lands wholly in one grid cell and pixel counts stay easy to reason about.
"""

import datetime as _dt
import pathlib as _pathlib

import numpy as np
import pytest

EPOCH = _dt.datetime(1987, 1, 1)
CADENCE = 3.8                      # seconds per low-resolution scan, as flown
NPIX_LO, NPIX_HI = 64, 128

# Channel positions in the Basist order the module bins into.
I19V, I19H, I22V, I37V, I37H, I85V, I85H = range(7)

LORES_VARS = ["ta19v", "ta19h", "ta22v", "ta37v", "ta37h"]
HIRES_VARS = ["ta85v", "ta85h"]

DEFAULT_TA = {"ta19v": 210.0, "ta19h": 200.0, "ta22v": 220.0, "ta37v": 230.0,
              "ta37h": 215.0, "ta85v": 240.0, "ta85h": 225.0}


def seconds(when):
    """Seconds since the product epoch, the units `xtime` carries."""
    return (when - EPOCH).total_seconds()


def day_start(date):
    """Seconds since the epoch at 00:00 UTC on a YYYYMMDD date."""
    return seconds(_dt.datetime.strptime(date, "%Y%m%d"))


def scans_in_day(start, nscan, date, cadence=CADENCE):
    """How many scans of a granule fall in a UTC day, computed independently."""
    t = seconds(start) + cadence * np.arange(nscan)
    t0 = day_start(date)
    return int(((t >= t0) & (t < t0 + 86400.0)).sum())


def write_granule(path, start, nscan=8, lat0=10.125, dlat=0.25, lon0=20.125,
                  values=None, qflag_lores=None, qflag_hires=None,
                  include_time=True, hires_scans=None, cadence=CADENCE,
                  lat_track=None, lon_step=0.0):
    """Write a minimal TDR BASE granule and return its path.

    `values` overrides individual channels, `hires_scans` breaks the two-to-one
    scan pairing, `include_time` drops `xtime` entirely, and `lat_track` sets an
    explicit scan-center latitude, for tracks that turn.

    `lon_step` spreads the pixels of a scan across track. Left at zero every
    pixel of a scan shares one cell, which keeps counts easy to reason about.
    Set it to a cell width or more to place pixels in distinct cells, which is
    what exercises cross-track cell assignment.
    """
    nc = pytest.importorskip("netCDF4")

    nhi = 2 * nscan if hires_scans is None else hires_scans
    track = (lat0 + dlat * np.arange(nscan)) if lat_track is None else \
        np.asarray(lat_track, float)
    across_lo = lon0 + lon_step * np.arange(NPIX_LO)
    across_hi = lon0 + lon_step * 0.5 * np.arange(NPIX_HI)
    lat_lo = np.repeat(track[:, None], NPIX_LO, 1)
    lon_lo = np.tile(across_lo, (nscan, 1))
    lat_hi = np.repeat((lat0 + dlat * 0.5 * np.arange(nhi))[:, None],
                       NPIX_HI, 1)
    lon_hi = np.tile(across_hi, (nhi, 1))
    values = dict(values or {})

    ds = nc.Dataset(path, "w", format="NETCDF4_CLASSIC")
    try:
        ds.createDimension("nscan_lores", nscan)
        ds.createDimension("npixel_lores", NPIX_LO)
        ds.createDimension("nscan_hires", nhi)
        ds.createDimension("npixel_hires", NPIX_HI)
        ds.createDimension("nchan_lores", 5)
        ds.createDimension("nchan_hires", 2)

        def put(name, dims, data, dtype="f4"):
            ds.createVariable(name, dtype, dims)[:] = data

        put("lat_lores", ("nscan_lores", "npixel_lores"), lat_lo)
        put("lon_lores", ("nscan_lores", "npixel_lores"), lon_lo)
        put("lat_hires", ("nscan_hires", "npixel_hires"), lat_hi)
        put("lon_hires", ("nscan_hires", "npixel_hires"), lon_hi)

        for name in LORES_VARS:
            put(name, ("nscan_lores", "npixel_lores"),
                values.get(name, np.full((nscan, NPIX_LO), DEFAULT_TA[name])))
        for name in HIRES_VARS:
            put(name, ("nscan_hires", "npixel_hires"),
                values.get(name, np.full((nhi, NPIX_HI), DEFAULT_TA[name])))

        put("qflag_lores", ("nscan_lores", "nchan_lores"),
            np.zeros((nscan, 5), np.int32) if qflag_lores is None
            else np.asarray(qflag_lores, np.int32), "i4")
        put("qflag_hires", ("nscan_hires", "nchan_hires"),
            np.zeros((nhi, 2), np.int32) if qflag_hires is None
            else np.asarray(qflag_hires, np.int32), "i4")

        if include_time:
            put("xtime", ("nscan_lores",),
                seconds(start) + cadence * np.arange(nscan), "f8")
    finally:
        ds.close()
    return path


def count_channel(counts, channel):
    """Pixels binned into a channel, summed over both passes and the globe."""
    return int(counts[:, :, :, channel].sum())


def write_full_day(path, date, **kw):
    """A granule whose scans span a whole UTC day with no gap.

    Real coverage needs a scan at least every `GAP_TOLERANCE` seconds, so this
    uses a 60 second cadence and 1440 scans rather than the flown 3.8 seconds,
    which would need about 22,700 scans and a fixture too large to be worth it.
    The latitude step is small enough that the track stays on the grid.
    """
    start = _dt.datetime.strptime(date, "%Y%m%d")
    kw.setdefault("lat0", -30.125)
    kw.setdefault("dlat", 0.04)
    return write_granule(path, start, nscan=1440, cadence=60.0, **kw)


FCDR_LORES = ["fcdr_tb19v", "fcdr_tb19h", "fcdr_tb22v", "fcdr_tb37v",
              "fcdr_tb37h"]
FCDR_HIRES = ["fcdr_tb85v", "fcdr_tb85h"]

# Brightness minus antenna, roughly the real per-channel offsets, so a paired
# fixture reads like the product it stands in for.
FCDR_OFFSET = {"fcdr_tb19v": 5.2, "fcdr_tb19h": 3.8, "fcdr_tb22v": 6.3,
               "fcdr_tb37v": 5.0, "fcdr_tb37h": 2.0, "fcdr_tb85v": 3.8,
               "fcdr_tb85h": 2.0}


def write_fcdr_granule(path, start, nscan=8, lat0=10.125, dlat=0.25,
                       lon0=20.125, values=None, quality_lores=None,
                       quality_hires=None, include_time=True,
                       hires_scans=None, cadence=CADENCE, lat_track=None,
                       lon_step=0.0, lat_shift=0.0):
    """Write a minimal FCDR swath granule, the counterpart of a TDR one.

    Geolocation defaults to the same coordinates as `write_granule` so a pair
    lines up, and `lat_shift` displaces it to stand in for the real product's
    geolocation correction. Quality is per pixel here, not per scan and channel,
    matching the real difference between the two products.
    """
    nc = pytest.importorskip("netCDF4")

    nhi = 2 * nscan if hires_scans is None else hires_scans
    track = (lat0 + dlat * np.arange(nscan)) if lat_track is None else \
        np.asarray(lat_track, float)
    track = track + lat_shift
    across_lo = lon0 + lon_step * np.arange(NPIX_LO)
    across_hi = lon0 + lon_step * 0.5 * np.arange(NPIX_HI)
    lat_lo = np.repeat(track[:, None], NPIX_LO, 1)
    lon_lo = np.tile(across_lo, (nscan, 1))
    lat_hi = np.repeat((lat0 + lat_shift + dlat * 0.5 * np.arange(nhi))[:, None],
                       NPIX_HI, 1)
    lon_hi = np.tile(across_hi, (nhi, 1))
    values = dict(values or {})

    ds = nc.Dataset(path, "w", format="NETCDF4_CLASSIC")
    try:
        ds.createDimension("nscan_lores", nscan)
        ds.createDimension("npixel_lores", NPIX_LO)
        ds.createDimension("nscan_hires", nhi)
        ds.createDimension("npixel_hires", NPIX_HI)

        def put(name, dims, data, dtype="f4"):
            ds.createVariable(name, dtype, dims)[:] = data

        put("lat_lores", ("nscan_lores", "npixel_lores"), lat_lo)
        put("lon_lores", ("nscan_lores", "npixel_lores"), lon_lo)
        put("lat_hires", ("nscan_hires", "npixel_hires"), lat_hi)
        put("lon_hires", ("nscan_hires", "npixel_hires"), lon_hi)

        for tb in FCDR_LORES:
            ta = DEFAULT_TA["ta" + tb[7:]]
            put(tb, ("nscan_lores", "npixel_lores"),
                values.get(tb, np.full((nscan, NPIX_LO), ta + FCDR_OFFSET[tb])))
        for tb in FCDR_HIRES:
            ta = DEFAULT_TA["ta" + tb[7:]]
            put(tb, ("nscan_hires", "npixel_hires"),
                values.get(tb, np.full((nhi, NPIX_HI), ta + FCDR_OFFSET[tb])))

        put("quality_lores", ("nscan_lores", "npixel_lores"),
            np.zeros((nscan, NPIX_LO), np.int8) if quality_lores is None
            else np.asarray(quality_lores, np.int8), "i1")
        put("quality_hires", ("nscan_hires", "npixel_hires"),
            np.zeros((nhi, NPIX_HI), np.int8) if quality_hires is None
            else np.asarray(quality_hires, np.int8), "i1")

        if include_time:
            put("scan_time_lores", ("nscan_lores",),
                seconds(start) + cadence * np.arange(nscan), "f8")
    finally:
        ds.close()
    return path


def write_pair(tmp_dir, rev, start, nscan=8, lat_shift=0.0, values=None,
               qflag_lores=None, qflag_hires=None, quality_lores=None,
               quality_hires=None, **shared):
    """A matched TDR and FCDR granule sharing an orbit revision number.

    Arguments that exist in only one product are routed to it. The two quality
    schemes are genuinely different, per scan and channel against per pixel, and
    `values` is split by name prefix so one call can set either arm's fields.
    """
    tmp_dir = _pathlib.Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    values = dict(values or {})
    ta_values = {k: v for k, v in values.items() if k.startswith("ta")}
    tb_values = {k: v for k, v in values.items() if k.startswith("fcdr_")}

    stem = (f"CSU_SSMI_{{}}_F13_D{start.strftime('%Y%m%d')}"
            f"_S{start:%H%M}_E0000_R{rev}.nc")
    tdr = write_granule(str(tmp_dir / stem.format("TDRBASE_V01R04")), start,
                        nscan=nscan, values=ta_values,
                        qflag_lores=qflag_lores, qflag_hires=qflag_hires,
                        **shared)
    fcdr = write_fcdr_granule(str(tmp_dir / stem.format("FCDR_V02R00")), start,
                              nscan=nscan, values=tb_values,
                              quality_lores=quality_lores,
                              quality_hires=quality_hires,
                              lat_shift=lat_shift, **shared)
    return tdr, fcdr
