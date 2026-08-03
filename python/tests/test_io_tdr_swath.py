"""Tests for the TDR swath reader and the daily antenna-temperature gridder.

The fixtures build minimal granules with the same variable names, shapes and
dimension names as the CSU TDR BASE product, so the reader is exercised through
its real code path without needing the archives, which are large and not in the
repository. One test reads an actual granule and is skipped when the 1998
download is not staged.

The gridder's central requirement is that a UTC day is assembled from the scans
that fall in it rather than from the archive that carries its date. CLASS files
a granule under the day it starts, so the orbit crossing midnight into a day
sits in the previous day's archive. Several tests below pin that behaviour,
since gridding a day from its own archive alone silently drops the first minutes
of every day.
"""

import datetime as _dt
import os

import numpy as np
import pytest

from swi import io_tdr_swath as tdr
from tdr_fixtures import (CADENCE, I19H, I19V, I85H, I85V, NPIX_HI, NPIX_LO,
                          count_channel as _count, scans_in_day, seconds
                          as _seconds, write_full_day, write_granule as _write_granule)


# --------------------------------------------------------------------------
# Grid geometry and cell indexing
# --------------------------------------------------------------------------

def test_grid_centers_match_the_csu_geometry():
    lat, lon = tdr.grid_centers()
    assert lat.size == 720 and lon.size == 1440
    assert lat[0] == pytest.approx(-89.875)
    assert lat[-1] == pytest.approx(89.875)
    assert lon[0] == pytest.approx(0.125)
    assert lon[-1] == pytest.approx(359.875)


def test_cell_index_places_corners_and_wraps_longitude():
    lat = np.array([-89.875, 89.875, 0.125, 0.125, np.nan, 95.0])
    lon = np.array([0.125, 359.875, -0.125, 360.125, 10.0, 10.0])
    ilat, ilon, ok = tdr._cell_index(lat, lon)

    assert ok[:4].all()
    assert not ok[4]                       # not finite
    assert not ok[5]                       # off the latitude grid
    assert (ilat[0], ilon[0]) == (0, 0)
    assert (ilat[1], ilon[1]) == (719, 1439)
    assert ilon[2] == 1439                 # -0.125 wraps to the last cell
    assert ilon[3] == 0                    # 360.125 wraps to the first


def test_cell_index_keeps_both_poles_and_rejects_beyond():
    """The latitude interval is closed at both ends.

    The last row spans 89.75 to 90.0, so a pixel at exactly 90.0 belongs to
    row 719 (from the geometry, (90 - -90) / 0.25 = 720 rows, indexed to 719),
    the same way -90.0 belongs to row 0. Flooring alone maps 90.0 to 720 and
    dropped it silently. Values past the pole and non-finite values must stay
    rejected, so the fold applies to the single boundary value only.
    """
    lat = np.array([90.0, -90.0, 90.0001, 89.9999, np.nan])
    lon = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    ilat, _, ok = tdr._cell_index(lat, lon)

    assert bool(ok[0]) and ilat[0] == 719  # exact north pole, last row
    assert bool(ok[1]) and ilat[1] == 0    # exact south pole, first row
    assert not ok[2]                       # beyond the pole stays rejected
    assert bool(ok[3]) and ilat[3] == 719
    assert not ok[4]                       # non-finite stays rejected


# --------------------------------------------------------------------------
# Scan-time selection, which is the defect this module had
# --------------------------------------------------------------------------

def test_scan_time_filter_splits_a_granule_across_the_two_days(tmp_path):
    """A granule spanning midnight contributes to each day only where it sits."""
    start = _dt.datetime(1998, 7, 15, 23, 59, 0)
    nscan = 40
    path = _write_granule(str(tmp_path / "g.nc"), start, nscan=nscan)

    # Independently: scans before midnight belong to the 15th.
    t = _seconds(start) + CADENCE * np.arange(nscan)
    midnight = _seconds(_dt.datetime(1998, 7, 16))
    n_15 = int((t < midnight).sum())
    n_16 = nscan - n_15
    assert n_15 > 0 and n_16 > 0           # the fixture really does cross

    _, _, _, c15, _ = tdr.grid_day([path], day="19980715")
    _, _, _, c16, _ = tdr.grid_day([path], day="19980716")
    _, _, _, call, _ = tdr.grid_day([path], day=None)

    assert _count(c15, I19V) == n_15 * NPIX_LO
    assert _count(c16, I19V) == n_16 * NPIX_LO
    assert _count(call, I19V) == nscan * NPIX_LO
    # The two days partition the granule, with nothing lost or double counted.
    assert _count(c15, I19V) + _count(c16, I19V) == _count(call, I19V)


def test_a_day_needs_the_previous_archive_to_be_complete(tmp_path):
    """The regression test for the defect.

    The granule crossing midnight is filed under the day it starts. Offering
    only the target day's own granules loses the first minutes of that day.
    """
    crossing = _write_granule(
        str(tmp_path / "cross.nc"), _dt.datetime(1998, 7, 15, 23, 59, 0),
        nscan=40)
    later = _write_granule(
        str(tmp_path / "later.nc"), _dt.datetime(1998, 7, 16, 2, 0, 0),
        nscan=40, lat0=30.125)

    _, _, _, own_only, rep_own = tdr.grid_day([later], day="19980716")
    _, _, _, with_prev, rep_both = tdr.grid_day([crossing, later],
                                                day="19980716")

    assert rep_own.used == [later]
    assert sorted(rep_both.used) == sorted([crossing, later])
    assert not rep_own.rejected and not rep_both.rejected
    assert not rep_own.empty and not rep_both.empty
    assert _count(with_prev, I19V) > _count(own_only, I19V)

    t = _seconds(_dt.datetime(1998, 7, 15, 23, 59, 0)) + CADENCE * np.arange(40)
    recovered = int((t >= _seconds(_dt.datetime(1998, 7, 16))).sum())
    assert _count(with_prev, I19V) - _count(own_only, I19V) == \
        recovered * NPIX_LO


def test_a_granule_outside_the_day_contributes_nothing(tmp_path):
    path = _write_granule(str(tmp_path / "g.nc"),
                          _dt.datetime(1998, 7, 20, 6, 0, 0), nscan=10)
    _, _, _, counts, report = tdr.grid_day([path], day="19980716")
    assert report.used == []
    assert report.rejected == []          # out of range is not a defect
    assert _count(counts, I19V) == 0


def test_a_granule_without_times_is_skipped_when_a_day_is_requested(tmp_path):
    """Binning it whole would import a foreign orbit into the day."""
    path = _write_granule(str(tmp_path / "notime.nc"),
                          _dt.datetime(1998, 7, 16, 2, 0, 0), nscan=10,
                          include_time=False)

    _, _, _, strict, report = tdr.grid_day([path], day="19980716")
    assert report.used == []
    assert _count(strict, I19V) == 0
    # The refusal is recorded, so the day cannot pass as whole.
    assert not report.complete
    assert report.rejected == [(path, "no usable scan times")]

    _, _, _, loose, loose_rep = tdr.grid_day([path], day="19980716",
                                             require_time=False)
    assert loose_rep.used == [path]
    assert _count(loose, I19V) == 10 * NPIX_LO
    assert not loose_rep.complete         # permissive, but still recorded


# --------------------------------------------------------------------------
# The two swaths
# --------------------------------------------------------------------------

def test_high_resolution_scans_inherit_the_low_resolution_day(tmp_path):
    """The 85 GHz swath carries no time of its own and pairs two to one."""
    start = _dt.datetime(1998, 7, 15, 23, 59, 0)
    nscan = 40
    path = _write_granule(str(tmp_path / "g.nc"), start, nscan=nscan)

    t = _seconds(start) + CADENCE * np.arange(nscan)
    n_15 = int((t < _seconds(_dt.datetime(1998, 7, 16))).sum())

    _, _, _, c15, _ = tdr.grid_day([path], day="19980715")
    _, _, _, c16, _ = tdr.grid_day([path], day="19980716")

    assert _count(c15, I85V) == 2 * n_15 * NPIX_HI
    assert _count(c16, I85V) == 2 * (nscan - n_15) * NPIX_HI
    assert _count(c15, I85H) == _count(c15, I85V)


def test_the_paired_high_resolution_scans_are_the_right_ones(tmp_path):
    """Counts alone do not pin the pairing.

    Selecting the wrong 85 GHz scans keeps the same total, so a test that checks
    only how many pixels were binned passes against a broken pairing. Giving
    each high-resolution scan its own grid row makes the identity of the kept
    scans visible.
    """
    start = _dt.datetime(1998, 7, 15, 23, 59, 0)
    nscan, lat0, dlat = 40, 10.125, 0.5      # 0.25 deg per hires scan, one row
    path = _write_granule(str(tmp_path / "g.nc"), start, nscan=nscan,
                          lat0=lat0, dlat=dlat)

    kept_15 = scans_in_day(start, nscan, "19980715")
    _, _, _, counts, _ = tdr.grid_day([path], day="19980716")

    rows = np.unique(np.argwhere(counts[:, :, :, I85V] > 0)[:, 1])
    first_row = int((lat0 + 90.0) / 0.25)
    # Two high-resolution scans per low-resolution scan, in file order, so the
    # day's scans are the contiguous block starting at twice the split point.
    expected = np.arange(first_row + 2 * kept_15, first_row + 2 * nscan)
    assert np.array_equal(rows, expected)
    # Contiguity is the property a tiled mask breaks: it would keep two blocks.
    assert rows.size == np.ptp(rows) + 1


def test_unpairable_high_resolution_swath_is_dropped_not_mixed(tmp_path):
    """A swath whose scans cannot be placed in a day must not be binned whole."""
    path = _write_granule(str(tmp_path / "odd.nc"),
                          _dt.datetime(1998, 7, 16, 2, 0, 0), nscan=10,
                          hires_scans=17)          # not two per low-res scan

    _, _, _, counts, report = tdr.grid_day([path], day="19980716")
    assert _count(counts, I19V) == 10 * NPIX_LO    # low-resolution still binned
    assert _count(counts, I85V) == 0               # high-resolution dropped
    assert report.used == [path]
    assert not report.complete
    assert len(report.rejected) == 1
    assert "do not pair" in report.rejected[0][1]


# --------------------------------------------------------------------------
# Masking and averaging
# --------------------------------------------------------------------------

def test_quality_flag_masks_only_its_own_channel(tmp_path):
    q = np.zeros((10, 5), np.int32)
    q[:, 1] = 1                                    # ta19h flagged
    qh = np.zeros((20, 2), np.int32)
    qh[:, 1] = 2                                   # ta85h flagged
    path = _write_granule(str(tmp_path / "q.nc"),
                          _dt.datetime(1998, 7, 16, 2, 0, 0), nscan=10,
                          qflag_lores=q, qflag_hires=qh)

    _, _, _, counts, _ = tdr.grid_day([path], day="19980716")
    assert _count(counts, I19H) == 0
    assert _count(counts, I85H) == 0
    assert _count(counts, I19V) == 10 * NPIX_LO
    assert _count(counts, I85V) == 20 * NPIX_HI


def test_unphysical_antenna_temperatures_are_excluded(tmp_path):
    ta = np.full((10, NPIX_LO), 210.0)
    ta[0, :] = 10.0                                # below the floor
    ta[1, :] = 500.0                               # above the ceiling
    ta[2, :] = np.nan
    path = _write_granule(str(tmp_path / "bad.nc"),
                          _dt.datetime(1998, 7, 16, 2, 0, 0), nscan=10,
                          values={"ta19v": ta})

    _, _, _, counts, _ = tdr.grid_day([path], day="19980716")
    assert _count(counts, I19V) == 7 * NPIX_LO
    assert _count(counts, I19H) == 10 * NPIX_LO


def test_a_cell_takes_the_mean_of_its_pixels(tmp_path):
    """Every pixel of a scan shares a cell, so the cell holds their mean."""
    nscan = 4
    ta = np.tile(np.arange(NPIX_LO, dtype=float), (nscan, 1)) + 200.0
    path = _write_granule(str(tmp_path / "m.nc"),
                          _dt.datetime(1998, 7, 16, 2, 0, 0), nscan=nscan,
                          dlat=0.25, values={"ta19v": ta})

    _, _, means, counts, _ = tdr.grid_day([path], day="19980716")
    filled = counts[:, :, :, I19V] > 0
    assert filled.sum() == nscan
    assert np.all(counts[:, :, :, I19V][filled] == NPIX_LO)
    assert means[:, :, :, I19V][filled] == pytest.approx(ta[0].mean())


def test_pixels_of_one_scan_spread_across_cells(tmp_path):
    """Cross-track placement, which a scan sharing one geolocation never tests."""
    path = _write_granule(str(tmp_path / "wide.nc"),
                          _dt.datetime(1998, 7, 16, 2, 0, 0), nscan=2,
                          lat_track=[10.125, 10.125], lon_step=0.25)

    _, _, _, counts, _ = tdr.grid_day([path], day="19980716")
    filled = counts[:, :, :, I19V] > 0
    # 64 pixels a quarter degree apart occupy 64 distinct cells, one row of them.
    assert filled.sum() == NPIX_LO
    assert np.unique(np.argwhere(filled)[:, 1]).size == 1       # a single row


def test_two_scans_landing_in_one_cell_are_averaged(tmp_path):
    """Cross-scan averaging within a cell, the other gap the flat fixture left."""
    ta = np.vstack([np.full(NPIX_LO, 200.0), np.full(NPIX_LO, 220.0)])
    path = _write_granule(str(tmp_path / "same.nc"),
                          _dt.datetime(1998, 7, 16, 2, 0, 0), nscan=2,
                          lat_track=[10.125, 10.125], lon_step=0.25,
                          values={"ta19v": ta})

    _, _, means, counts, _ = tdr.grid_day([path], day="19980716")
    filled = counts[:, :, :, I19V] > 0
    assert filled.sum() == NPIX_LO
    assert np.all(counts[:, :, :, I19V][filled] == 2)           # both scans
    assert means[:, :, :, I19V][filled] == pytest.approx(210.0)


def test_the_same_file_offered_twice_is_read_once(tmp_path):
    """Otherwise an overlapping pair of archives would double weight a granule."""
    path = _write_granule(str(tmp_path / "g.nc"),
                          _dt.datetime(1998, 7, 16, 2, 0, 0), nscan=10)

    _, _, _, once, _ = tdr.grid_day([path], day="19980716")
    _, _, _, twice, report = tdr.grid_day([path, path], day="19980716")

    assert _count(twice, I19V) == _count(once, I19V)
    assert report.used == [path]


def test_empty_cells_are_not_a_number(tmp_path):
    path = _write_granule(str(tmp_path / "g.nc"),
                          _dt.datetime(1998, 7, 16, 2, 0, 0), nscan=4)
    _, _, means, counts, _ = tdr.grid_day([path], day="19980716")
    empty = counts[:, :, :, I19V] == 0
    assert empty.any()
    assert np.isnan(means[:, :, :, I19V][empty]).all()


# --------------------------------------------------------------------------
# Ascending and descending separation
# --------------------------------------------------------------------------

def test_ascending_and_descending_tracks_separate(tmp_path):
    up = _write_granule(str(tmp_path / "up.nc"),
                        _dt.datetime(1998, 7, 16, 2, 0, 0), nscan=10,
                        dlat=0.25)
    down = _write_granule(str(tmp_path / "down.nc"),
                          _dt.datetime(1998, 7, 16, 4, 0, 0), nscan=10,
                          lat0=40.125, dlat=-0.25)

    _, _, _, cu, _ = tdr.grid_day([up], day="19980716")
    _, _, _, cd, _ = tdr.grid_day([down], day="19980716")

    assert cu[0, :, :, I19V].sum() == 10 * NPIX_LO and cu[1, :, :, I19V].sum() == 0
    assert cd[1, :, :, I19V].sum() == 10 * NPIX_LO and cd[0, :, :, I19V].sum() == 0


def _orbit_track(namp=0.02, n=240):
    """Scan-center latitude over one turn, flattening at the extremes as flown."""
    rng = np.random.default_rng(0)
    phase = np.linspace(-np.pi / 2, 3 * np.pi / 2, n)
    return 80.0 * np.sin(phase) + rng.normal(0.0, namp, n)


def test_orbital_turning_point_gives_one_clean_transition():
    """Noise where the track flattens must not flip scans into the wrong pass."""
    asc = tdr._ascending(_orbit_track())
    flips = int((asc[1:] != asc[:-1]).sum())
    assert flips == 1, "the smoothed gradient should turn exactly once"
    assert asc[:100].all()
    assert not asc[-100:].any()


def test_raw_gradient_would_have_flipped_at_the_turn():
    """States the problem the smoothing solves, so a regression stays visible."""
    track = _orbit_track()
    naive = np.gradient(track) > 0
    assert int((naive[1:] != naive[:-1]).sum()) > 1
    asc = tdr._ascending(track)
    assert int((asc[1:] != asc[:-1]).sum()) == 1


def test_pass_is_computed_before_the_day_filter_is_applied(tmp_path,
                                                           monkeypatch):
    """Pins the ordering directly, by watching what _ascending is handed.

    Comparing the resulting pass split does not discriminate here, because for a
    long enough track the smoothed gradient reaches the same answer whether it
    sees the whole granule or only the part inside the day. Only the argument
    _ascending receives distinguishes the two orderings.
    """
    start = _dt.datetime(1998, 7, 15, 23, 59, 0)
    nscan = 60
    path = _write_granule(str(tmp_path / "g.nc"), start, nscan=nscan)

    in_day = scans_in_day(start, nscan, "19980716")
    assert 0 < in_day < nscan              # the granule really is split

    real = tdr._ascending
    seen = []

    def spy(track):
        seen.append(track.size)
        return real(track)

    monkeypatch.setattr(tdr, "_ascending", spy)
    tdr.grid_day([path], day="19980716")

    # The whole track of each swath, not the 44 scans that fall inside the day.
    assert seen == [nscan, 2 * nscan]


def test_pass_split_of_a_turning_granule_matches_the_full_track(tmp_path):
    """Filtering by day must not change which pass a scan is assigned to."""
    start = _dt.datetime(1998, 7, 15, 23, 59, 0)
    nscan = 60
    lat0, dlat = 10.125, 0.25
    # A track that turns inside the granule, after midnight.
    lat = lat0 + dlat * np.arange(nscan)
    lat[45:] = lat[44] - dlat * np.arange(1, nscan - 44)

    path = _write_granule(str(tmp_path / "turn.nc"), start, nscan=nscan,
                          lat_track=lat)

    t = _seconds(start) + CADENCE * np.arange(nscan)
    in_16 = t >= _seconds(_dt.datetime(1998, 7, 16))
    expected_asc = int((tdr._ascending(lat) & in_16).sum())
    expected_dsc = int((~tdr._ascending(lat) & in_16).sum())
    assert expected_asc > 0 and expected_dsc > 0   # the turn is in this day

    _, _, _, counts, _ = tdr.grid_day([path], day="19980716")
    assert counts[0, :, :, I19V].sum() == expected_asc * NPIX_LO
    assert counts[1, :, :, I19V].sum() == expected_dsc * NPIX_LO


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

def test_used_lists_only_the_granules_that_contributed(tmp_path):
    inside = _write_granule(str(tmp_path / "in.nc"),
                            _dt.datetime(1998, 7, 16, 2, 0, 0), nscan=10)
    outside = _write_granule(str(tmp_path / "out.nc"),
                             _dt.datetime(1998, 7, 20, 2, 0, 0), nscan=10)

    _, _, _, report = tdr.grid_day([outside, inside], day="19980716")[1:]
    assert report.used == [inside]


def test_write_grid_records_only_contributing_granules(tmp_path):
    nc = pytest.importorskip("netCDF4")
    inside = _write_granule(str(tmp_path / "in.nc"),
                            _dt.datetime(1998, 7, 16, 2, 0, 0), nscan=10)
    outside = _write_granule(str(tmp_path / "out.nc"),
                             _dt.datetime(1998, 7, 20, 2, 0, 0), nscan=10)

    lat, lon, means, counts, report = tdr.grid_day([outside, inside],
                                                   day="19980716")
    out = str(tmp_path / "grid.nc")
    tdr.write_grid(out, lat, lon, means, counts, report.used, "F13",
                   report=report)

    ds = nc.Dataset(out)
    try:
        assert ds.source_granules == "in.nc"
        assert ds.satellite == "F13"
        assert ds["ta19v_asc"].shape == (720, 1440)
        binned = np.asarray(ds["n_ta19v_asc"][:])
        assert binned.sum() == 10 * NPIX_LO
    finally:
        ds.close()


def test_a_missing_quality_flag_refuses_the_swath(tmp_path):
    """Unfiltered values look like good data, so they must not pass silently."""
    nc = pytest.importorskip("netCDF4")
    path = _write_granule(str(tmp_path / "noq.nc"),
                          _dt.datetime(1998, 7, 16, 2, 0, 0), nscan=10)
    # Drop the low-resolution quality flag by rewriting without it.
    src = nc.Dataset(path)
    keep = {n: (np.asarray(v[:]), v.dimensions, v.dtype)
            for n, v in src.variables.items() if n != "qflag_lores"}
    dims = {n: d.size for n, d in src.dimensions.items()}
    src.close()
    out = nc.Dataset(path, "w", format="NETCDF4_CLASSIC")
    for n, size in dims.items():
        out.createDimension(n, size)
    for n, (data, d, dt) in keep.items():
        out.createVariable(n, dt, d)[:] = data
    out.close()

    _, _, _, counts, report = tdr.grid_day([path], day="19980716")
    assert _count(counts, I19V) == 0            # low-resolution refused
    assert _count(counts, I85V) == 20 * NPIX_HI  # high-resolution still fine
    assert not report.complete
    assert "quality flags unusable" in report.rejected[0][1]


def test_an_interrupted_write_leaves_the_previous_day_intact(tmp_path,
                                                             monkeypatch):
    """A rebuild must not truncate a good file when it fails part way."""
    nc = pytest.importorskip("netCDF4")
    path = _write_granule(str(tmp_path / "g.nc"),
                          _dt.datetime(1998, 7, 16, 2, 0, 0), nscan=10)
    lat, lon, means, counts, report = tdr.grid_day([path], day="19980716")
    out = str(tmp_path / "day.nc")
    tdr.write_grid(out, lat, lon, means, counts, report.used, "F13")
    good = os.path.getsize(out)

    def boom(*a, **k):
        raise RuntimeError("interrupted part way through")

    monkeypatch.setattr(tdr, "_fill_grid", boom)
    with pytest.raises(RuntimeError):
        tdr.write_grid(out, lat, lon, means, counts, report.used, "F13")

    assert os.path.getsize(out) == good          # the old day survives
    assert not os.path.exists(out + ".partial")  # and nothing is left over
    ds = nc.Dataset(out)
    try:
        assert ds.satellite == "F13"             # still readable
    finally:
        ds.close()


def test_a_full_day_with_nothing_refused_is_marked_complete(tmp_path):
    nc = pytest.importorskip("netCDF4")
    path = write_full_day(str(tmp_path / "whole.nc"), "19980716")

    lat, lon, means, counts, report = tdr.grid_day([path], day="19980716")
    assert report.complete
    assert report.gaps == []
    assert report.covered == pytest.approx(86400.0)

    out = str(tmp_path / "whole_grid.nc")
    tdr.write_grid(out, lat, lon, means, counts, report.used, "F13",
                   report=report)
    ds = nc.Dataset(out)
    try:
        assert ds.complete == "true"
        assert ds.coverage_fraction == pytest.approx(1.0)
        assert ds.coverage_gaps_utc == ""
    finally:
        ds.close()


def test_a_hole_in_the_scan_times_makes_the_day_incomplete(tmp_path):
    """No input is refused here. Only the measured coverage reveals the loss."""
    nc = pytest.importorskip("netCDF4")
    early = write_full_day(str(tmp_path / "early.nc"), "19980716")
    # Keep the first six hours and the last six, dropping the middle twelve.
    src = nc.Dataset(early, "a")
    x = np.asarray(src["xtime"][:], float)
    t0 = tdr._day_start("19980716")
    hole = (x >= t0 + 6 * 3600) & (x < t0 + 18 * 3600)
    ta = np.asarray(src["ta19v"][:], float)
    for name in ("ta19v", "ta19h", "ta22v", "ta37v", "ta37h"):
        v = np.asarray(src[name][:], float)
        v[hole, :] = np.nan
        src[name][:] = v
    src["xtime"][:] = np.where(hole, t0 - 1e6, x)   # push them out of the day
    src.close()

    _, _, _, _, report = tdr.grid_day([early], day="19980716")
    assert not report.complete
    assert report.rejected == []                   # nothing was refused
    assert len(report.gaps) == 1
    start, end = report.gaps[0]
    assert start == pytest.approx(6 * 3600, abs=120)
    assert end == pytest.approx(18 * 3600, abs=120)
    assert report.covered_fraction == pytest.approx(0.5, abs=0.01)
    del ta


def test_a_granule_whose_pixels_all_fail_screening_is_recorded(tmp_path):
    """It contributes nothing yet is refused by nothing, so it needs its own list."""
    good = write_full_day(str(tmp_path / "good.nc"), "19980716")
    q = np.zeros((10, 5), np.int32) + 1            # every low-res channel flagged
    qh = np.zeros((20, 2), np.int32) + 1
    bad = _write_granule(str(tmp_path / "flagged.nc"),
                         _dt.datetime(1998, 7, 16, 3, 0, 0), nscan=10,
                         lat0=40.125, qflag_lores=q, qflag_hires=qh)

    _, _, _, _, report = tdr.grid_day([good, bad], day="19980716")
    assert report.used == [good]
    assert report.rejected == []
    assert len(report.empty) == 1
    assert report.empty[0][0] == bad
    assert "no usable pixels" in report.empty[0][1]
    assert not report.complete


def test_a_refusal_is_named_in_the_written_file(tmp_path):
    nc = pytest.importorskip("netCDF4")
    good = write_full_day(str(tmp_path / "good.nc"), "19980716")
    broken = _write_granule(str(tmp_path / "bad.nc"),
                            _dt.datetime(1998, 7, 16, 4, 0, 0), nscan=10,
                            lat0=40.125, hires_scans=17)

    lat, lon, means, counts, report = tdr.grid_day([good, broken],
                                                   day="19980716")
    out = str(tmp_path / "grid.nc")
    tdr.write_grid(out, lat, lon, means, counts, report.used, "F13",
                   report=report)
    ds = nc.Dataset(out)
    try:
        assert ds.complete == "false"
        assert "bad.nc" in ds.rejected_granules
    finally:
        ds.close()


# --------------------------------------------------------------------------
# Reading failures must cost one granule, not the whole run
# --------------------------------------------------------------------------

def test_an_unreadable_granule_is_refused_not_raised(tmp_path):
    good = write_full_day(str(tmp_path / "good.nc"), "19980716")
    junk = str(tmp_path / "corrupt.nc")
    with open(junk, "wb") as fh:
        fh.write(b"this is not a netCDF file at all")

    _, _, _, counts, report = tdr.grid_day([good, junk], day="19980716")
    assert report.used == [good]                   # the good day still builds
    assert _count(counts, I19V) > 0
    assert len(report.rejected) == 1
    assert report.rejected[0][0] == junk
    assert "unreadable" in report.rejected[0][1]
    assert not report.complete


def test_a_granule_missing_a_channel_is_refused_not_raised(tmp_path):
    nc = pytest.importorskip("netCDF4")
    good = write_full_day(str(tmp_path / "good.nc"), "19980716")
    maimed = _write_granule(str(tmp_path / "maimed.nc"),
                            _dt.datetime(1998, 7, 16, 5, 0, 0), nscan=10,
                            lat0=45.125)
    src = nc.Dataset(maimed)
    keep = {n: (np.asarray(v[:]), v.dimensions, v.dtype)
            for n, v in src.variables.items() if n != "ta37v"}
    dims = {n: d.size for n, d in src.dimensions.items()}
    src.close()
    out = nc.Dataset(maimed, "w", format="NETCDF4_CLASSIC")
    for n, size in dims.items():
        out.createDimension(n, size)
    for n, (data, d, dt) in keep.items():
        out.createVariable(n, dt, d)[:] = data
    out.close()

    _, _, _, _, report = tdr.grid_day([good, maimed], day="19980716")
    assert report.used == [good]
    assert len(report.rejected) == 1
    assert report.rejected[0][0] == maimed
    assert not report.complete


# --------------------------------------------------------------------------
# An actual granule, when the 1998 download is staged
# --------------------------------------------------------------------------

REAL = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                    "f13_1998_tdr",
                    "CSU_SSMI_TDRBASE_V01R04_F13_D19980715.tar")


@pytest.mark.skipif(not os.path.exists(REAL),
                    reason="July 1998 F-13 TDR archive not staged")
def test_real_crossing_granule_belongs_almost_entirely_to_the_next_day():
    """The archive's last granule holds 1585 of its 1605 scans on the 16th.

    Selecting granules by the date in the archive name rather than by scan time
    therefore lost most of an orbit from every day, not merely a tail.
    """
    pytest.importorskip("netCDF4")
    import tarfile
    import tempfile

    with tempfile.TemporaryDirectory() as work:
        with tarfile.open(REAL) as tf:
            last = sorted(m for m in tf.getnames() if m.endswith(".nc"))[-1]
            try:
                tf.extract(last, work, filter="data")
            except TypeError:                      # Python < 3.12
                tf.extract(last, work)
        path = os.path.join(work, last)
        assert "_S2358_E0140_" in os.path.basename(path)

        acc15 = tdr._Accumulator()
        acc16 = tdr._Accumulator()
        r15 = tdr.add_granule(acc15, path, day="19980715")
        r16 = tdr.add_granule(acc16, path, day="19980716")

    assert r15.kept == 20 and r16.kept == 1585      # the split, read from xtime
    n15, n16 = r15.binned, r16.binned
    assert n15 > 0 and n16 > 0
    # 20 scans on the 15th against 1585 on the 16th, so the 16th holds the bulk.
    assert n16 > 50 * n15


def test_the_accumulator_refuses_an_index_outside_the_grid():
    """numpy applies a negative flat index silently from the array's end, so
    the in-grid property must be asserted, not assumed. This pins the backstop
    for any caller that skips the _cell_index mask."""
    acc = tdr._Accumulator()
    with pytest.raises(AssertionError):
        acc.add(0, 0, np.array([-1]), np.array([0]), np.array([200.0]))
    with pytest.raises(AssertionError):
        acc.add(0, 0, np.array([tdr.NLAT]), np.array([0]), np.array([200.0]))
