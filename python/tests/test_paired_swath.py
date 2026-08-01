"""Tests for the paired antenna and brightness temperature gridder.

The property that matters is that the two arms differ in exactly one respect.
Everything else, the geolocation, the scan times, the quality screening, the
pass assignment and the contributing pixels of every cell, must be identical, or
the comparison carries the processing chain as well as the input convention.
Most of these tests exist to pin that, and several would pass trivially if the
two arms were simply built from the same file, so each one is written to fail
when a specific thing is allowed to differ.
"""

import datetime as _dt
import os

import numpy as np
import pytest

from swi import paired_swath as ps
from tdr_fixtures import (I19V, I85V, NPIX_HI, NPIX_LO, write_fcdr_granule,
                          write_granule, write_pair)

START = _dt.datetime(1998, 7, 16, 2, 0, 0)
DAY = "19980716"


def _grid(pairs, day=DAY):
    return ps.grid_day_paired(pairs, day=day)


# --------------------------------------------------------------------------
# Matching granules by orbit
# --------------------------------------------------------------------------

def test_revision_is_read_from_the_filename():
    assert ps.revision("CSU_SSMI_FCDR_V02R00_F13_D19980715_S2358_E0140_"
                       "R017078.nc") == "017078"
    assert ps.revision("nothing_here.nc") is None


def test_granules_pair_by_orbit_and_strays_are_reported(tmp_path):
    a_t, a_f = write_pair(tmp_path, "017001", START)
    b_t, b_f = write_pair(tmp_path, "017002",
                          START + _dt.timedelta(hours=2))
    lonely = write_granule(str(tmp_path / "solo_R017003.nc"),
                           START + _dt.timedelta(hours=4))

    pairs, unmatched = ps.pair_paths([a_t, b_t, lonely], [a_f, b_f])
    assert [r for r, _, _ in pairs] == ["017001", "017002"]
    assert unmatched == [lonely]


def test_a_granule_present_in_one_product_only_is_never_silently_used(tmp_path):
    """One-sided coverage is the bias this whole design exists to avoid."""
    a_t, a_f = write_pair(tmp_path, "017001", START)
    orphan = write_fcdr_granule(str(tmp_path / "orphan_R017009.nc"),
                                START + _dt.timedelta(hours=3))

    pairs, unmatched = ps.pair_paths([a_t], [a_f, orphan])
    assert len(pairs) == 1
    assert unmatched == [orphan]
    # The orphan reaches no accumulator, so neither arm gains coverage from it.
    _, _, _, _, counts, report = _grid(pairs)
    assert report.used == [a_t]


# --------------------------------------------------------------------------
# The single-operator guarantee
# --------------------------------------------------------------------------

def test_both_arms_rest_on_identical_contributing_pixels(tmp_path):
    t, f = write_pair(tmp_path, "017001", START, nscan=20)
    _, _, means_ta, means_tb, counts, _ = _grid([("017001", t, f)])

    filled = counts[:, :, :, I19V] > 0
    assert filled.sum() > 0
    # Defined in both arms exactly where the shared count is positive.
    assert np.array_equal(np.isfinite(means_ta[:, :, :, I19V]), filled)
    assert np.array_equal(np.isfinite(means_tb[:, :, :, I19V]), filled)


def test_the_fcdr_geolocation_places_both_arms(tmp_path):
    """The products disagree on position, so one of them must win, and it is
    the corrected one. Shifting only the FCDR coordinates must move both arms.
    """
    plain_t, plain_f = write_pair(tmp_path / "a", "017001", START, nscan=8)
    _, _, _, _, counts_plain, _ = _grid([("017001", plain_t, plain_f)])
    rows_plain = np.unique(np.argwhere(counts_plain[:, :, :, I19V] > 0)[:, 1])

    shifted_t, shifted_f = write_pair(tmp_path / "b", "017001", START, nscan=8,
                                      lat_shift=2.0)
    _, _, _, _, counts_shifted, _ = _grid([("017001", shifted_t, shifted_f)])
    rows_shifted = np.unique(
        np.argwhere(counts_shifted[:, :, :, I19V] > 0)[:, 1])

    # 2 degrees is 8 cells at a quarter degree, and the TDR file is unmoved.
    assert np.array_equal(rows_shifted, rows_plain + 8)


def test_a_pixel_flagged_by_either_product_is_dropped_from_both(tmp_path):
    """The two screens differ in shape, so neither alone would screen both arms."""
    nscan = 10
    # FCDR flags the first three scans, per pixel.
    q = np.zeros((nscan, NPIX_LO), np.int8)
    q[:3, :] = 14
    t_fcdr_flag, f_fcdr_flag = write_pair(tmp_path / "a", "017001", START,
                                          nscan=nscan, quality_lores=q)
    _, _, ta, tb, counts, _ = _grid([("017001", t_fcdr_flag, f_fcdr_flag)])
    assert counts[:, :, :, I19V].sum() == (nscan - 3) * NPIX_LO

    # TDR flags one channel across every scan, per scan and channel.
    qf = np.zeros((nscan, 5), np.int32)
    qf[:, I19V] = 1
    t_tdr_flag, f_tdr_flag = write_pair(tmp_path / "b", "017002", START,
                                        nscan=nscan, qflag_lores=qf)
    _, _, _, _, counts2, _ = _grid([("017002", t_tdr_flag, f_tdr_flag)])
    assert counts2[:, :, :, I19V].sum() == 0          # dropped from both arms
    assert counts2[:, :, :, 1].sum() == nscan * NPIX_LO   # 19H untouched


def test_a_pixel_unphysical_in_one_arm_is_dropped_from_both(tmp_path):
    """Otherwise a cell would rest on different pixels in the two arms."""
    nscan = 10
    tb = np.full((nscan, NPIX_LO), 215.2)
    tb[0, :] = 1e4                                     # unphysical brightness
    t, f = write_pair(tmp_path, "017001", START, nscan=nscan,
                      values={"fcdr_tb19v": tb})

    _, _, means_ta, means_tb, counts, _ = _grid([("017001", t, f)])
    assert counts[:, :, :, I19V].sum() == (nscan - 1) * NPIX_LO
    filled = counts[:, :, :, I19V] > 0
    # The antenna arm is finite exactly where the brightness arm is.
    assert np.array_equal(np.isfinite(means_ta[:, :, :, I19V]), filled)
    assert np.array_equal(np.isfinite(means_tb[:, :, :, I19V]), filled)


def test_the_recovered_offsets_are_the_ones_written_in(tmp_path):
    t, f = write_pair(tmp_path, "017001", START, nscan=12)
    _, _, means_ta, means_tb, counts, _ = _grid([("017001", t, f)])

    for ch, expected in ((I19V, 5.2), (I85V, 3.8)):
        filled = counts[:, :, :, ch] > 0
        diff = (means_tb - means_ta)[:, :, :, ch][filled]
        assert diff == pytest.approx(expected, abs=1e-3)


# --------------------------------------------------------------------------
# Day assembly and refusals
# --------------------------------------------------------------------------

def test_scan_times_come_from_the_fcdr_and_split_the_day(tmp_path):
    cross = _dt.datetime(1998, 7, 15, 23, 59, 0)
    t, f = write_pair(tmp_path, "017001", cross, nscan=40)

    _, _, _, _, c15, _ = _grid([("017001", t, f)], day="19980715")
    _, _, _, _, c16, _ = _grid([("017001", t, f)], day="19980716")
    total = c15[:, :, :, I19V].sum() + c16[:, :, :, I19V].sum()
    assert c15[:, :, :, I19V].sum() > 0 and c16[:, :, :, I19V].sum() > 0
    assert total == 40 * NPIX_LO           # partitioned, nothing lost or doubled


def test_mismatched_scan_counts_are_refused(tmp_path):
    t = write_granule(str(tmp_path / "t_R017001.nc"), START, nscan=10)
    f = write_fcdr_granule(str(tmp_path / "f_R017001.nc"), START, nscan=9)

    _, _, _, _, counts, report = _grid([("017001", t, f)])
    assert counts.sum() == 0
    assert len(report.rejected) == 1
    assert "scan counts differ" in report.rejected[0][1]


def test_an_unreadable_counterpart_is_refused_not_raised(tmp_path):
    good_t, good_f = write_pair(tmp_path, "017001", START, nscan=10)
    bad_t = write_granule(str(tmp_path / "bad_R017002.nc"),
                          START + _dt.timedelta(hours=2), nscan=10)
    junk = str(tmp_path / "junk_R017002.nc")
    with open(junk, "wb") as fh:
        fh.write(b"not netCDF")

    _, _, _, _, counts, report = _grid([("017001", good_t, good_f),
                                        ("017002", bad_t, junk)])
    assert counts[:, :, :, I19V].sum() == 10 * NPIX_LO   # the good pair survives
    assert report.used == [good_t]
    assert any("unreadable" in r for _, r in report.rejected)
    assert not report.complete


def test_the_high_resolution_arms_stay_paired_across_midnight(tmp_path):
    cross = _dt.datetime(1998, 7, 15, 23, 59, 0)
    t, f = write_pair(tmp_path, "017001", cross, nscan=40)
    _, _, _, _, c16, _ = _grid([("017001", t, f)], day="19980716")

    # Two high-resolution scans per low-resolution scan, both arms alike.
    lo = c16[:, :, :, I19V].sum() / NPIX_LO
    hi = c16[:, :, :, I85V].sum() / NPIX_HI
    assert hi == 2 * lo
