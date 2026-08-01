"""Tests for the daily antenna-temperature gridding driver.

The driver's job is to decide which granules a UTC day is built from. CLASS
files a granule under the day it starts, so the orbit that crosses midnight into
a day sits in the previous day's archive, and a driver that hands the gridder
only the archive whose name carries the target date silently drops the first
minutes of every day. These tests pin the selection, the reporting of days built
without a predecessor, and the provenance written into the output.
"""

import datetime as _dt
import os
import tarfile

import numpy as np
import pytest

from scripts import grid_tdr_day as drv
from tdr_fixtures import NPIX_LO, scans_in_day, write_granule

CROSS_START = _dt.datetime(1998, 7, 15, 23, 59, 0)
LATER_START = _dt.datetime(1998, 7, 16, 2, 0, 0)
NSCAN = 40

DIR_15 = "CSU_SSMI_TDRBASE_V01R04_F13_D19980715"
DIR_16 = "CSU_SSMI_TDRBASE_V01R04_F13_D19980716"
CROSS_NC = "CSU_SSMI_TDRBASE_V01R04_F13_D19980715_S2359_E0102.nc"
LATER_NC = "CSU_SSMI_TDRBASE_V01R04_F13_D19980716_S0200_E0342.nc"


def _archives(tmp_path):
    """Two day directories, the first holding the granule that crosses midnight."""
    d15 = tmp_path / DIR_15
    d16 = tmp_path / DIR_16
    d15.mkdir()
    d16.mkdir()
    write_granule(str(d15 / CROSS_NC), CROSS_START, nscan=NSCAN)
    write_granule(str(d16 / LATER_NC), LATER_START, nscan=NSCAN, lat0=30.125)
    return str(d15), str(d16)


def _run(monkeypatch, args):
    monkeypatch.setattr("sys.argv", ["grid_tdr_day"] + args)
    return drv.main()


def _read(path, var="n_ta19v_asc"):
    nc = pytest.importorskip("netCDF4")
    ds = nc.Dataset(path)
    try:
        return int(np.asarray(ds[var][:]).sum()), ds.source_granules
    finally:
        ds.close()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def test_day_key_reads_the_satellite_and_date():
    assert drv.day_key("CSU_SSMI_TDRBASE_V01R04_F13_D19980715.tar") == \
        ("F13", "19980715")
    assert drv.day_key("/a/b/CSU_x_F16_D20160701_S0000_E0142.nc") == \
        ("F16", "20160701")
    assert drv.day_key("no_date_here.tar") is None


def test_shift_day_crosses_month_and_year_boundaries():
    assert drv.shift_day("19980715", -1) == "19980714"
    assert drv.shift_day("19980701", -1) == "19980630"
    assert drv.shift_day("19980731", 1) == "19980801"
    assert drv.shift_day("19990101", -1) == "19981231"
    assert drv.shift_day("19960301", -1) == "19960229"      # a leap year


# --------------------------------------------------------------------------
# Granule selection, which is the defect this driver had
# --------------------------------------------------------------------------

def test_a_day_is_built_from_the_neighbouring_archives(tmp_path, monkeypatch):
    d15, d16 = _archives(tmp_path)
    out = str(tmp_path / "out")

    assert _run(monkeypatch, [d15, d16, "--out", out]) == 0

    carried = scans_in_day(CROSS_START, NSCAN, "19980716")
    kept_15 = scans_in_day(CROSS_START, NSCAN, "19980715")
    assert carried > 0 and kept_15 > 0        # the fixture really does cross

    n16, sources16 = _read(os.path.join(out, "SWI_TDRGRID_F13_D19980716.nc"))
    n15, sources15 = _read(os.path.join(out, "SWI_TDRGRID_F13_D19980715.nc"))

    # The 16th picks up the tail of the 15th's archive as well as its own orbit.
    assert n16 == (carried + NSCAN) * NPIX_LO
    assert CROSS_NC in sources16 and LATER_NC in sources16
    # The 15th keeps only the scans that fall before midnight.
    assert n15 == kept_15 * NPIX_LO
    assert sources15 == CROSS_NC


def test_a_day_without_its_predecessor_is_reported_as_short(tmp_path,
                                                            monkeypatch,
                                                            capsys):
    _, d16 = _archives(tmp_path)
    out = str(tmp_path / "out")

    assert _run(monkeypatch, [d16, "--out", out]) == 0
    printed = capsys.readouterr().out

    n16, sources16 = _read(os.path.join(out, "SWI_TDRGRID_F13_D19980716.nc"))
    assert n16 == NSCAN * NPIX_LO             # short by the carried scans
    assert CROSS_NC not in sources16
    assert "SHORT" in printed
    assert "19980716" in printed


def test_the_short_day_is_the_one_the_old_selection_produced(tmp_path,
                                                             monkeypatch):
    """The two runs differ by exactly the scans the crossing granule carries."""
    d15, d16 = _archives(tmp_path)
    full = str(tmp_path / "full")
    short = str(tmp_path / "short")

    _run(monkeypatch, [d15, d16, "--out", full])
    _run(monkeypatch, [d16, "--out", short])

    n_full, _ = _read(os.path.join(full, "SWI_TDRGRID_F13_D19980716.nc"))
    n_short, _ = _read(os.path.join(short, "SWI_TDRGRID_F13_D19980716.nc"))
    carried = scans_in_day(CROSS_START, NSCAN, "19980716")
    assert n_full - n_short == carried * NPIX_LO


def test_a_granule_that_contributes_nothing_is_not_recorded(tmp_path,
                                                            monkeypatch):
    """Neighbouring archives are offered, so provenance must list contributors."""
    d15, d16 = _archives(tmp_path)
    d17 = tmp_path / "CSU_SSMI_TDRBASE_V01R04_F13_D19980717"
    d17.mkdir()
    far = "CSU_SSMI_TDRBASE_V01R04_F13_D19980717_S1200_E1342.nc"
    write_granule(str(d17 / far), _dt.datetime(1998, 7, 17, 12, 0, 0),
                  nscan=NSCAN, lat0=50.125)
    out = str(tmp_path / "out")

    _run(monkeypatch, [d15, d16, str(d17), "--out", out])

    _, sources16 = _read(os.path.join(out, "SWI_TDRGRID_F13_D19980716.nc"))
    assert far not in sources16               # offered as a neighbour, unused
    assert LATER_NC in sources16


# --------------------------------------------------------------------------
# Several satellites, and overlapping archives
# --------------------------------------------------------------------------

def test_one_satellite_never_reaches_into_another_satellites_grid(tmp_path,
                                                                  monkeypatch):
    """Neighbouring archives are offered, so the satellite has to be matched.

    Keying the archive map on the date alone let an F-13 orbit be binned into an
    F-16 day, and the output still carried the F-16 label.
    """
    d13 = tmp_path / "CSU_SSMI_TDRBASE_V01R04_F13_D19980715"
    d16 = tmp_path / "CSU_SSMI_TDRBASE_V01R04_F16_D19980716"
    d13.mkdir()
    d16.mkdir()
    # Distinct constant temperatures make the origin of every pixel legible.
    write_granule(str(d13 / "f13_cross.nc"), CROSS_START, nscan=NSCAN,
                  values={"ta19v": np.full((NSCAN, NPIX_LO), 111.0)})
    write_granule(str(d16 / "f16_own.nc"), LATER_START, nscan=NSCAN,
                  lat0=30.125,
                  values={"ta19v": np.full((NSCAN, NPIX_LO), 222.0)})
    out = str(tmp_path / "out")

    assert _run(monkeypatch, [str(d13), str(d16), "--out", out]) == 0

    nc = pytest.importorskip("netCDF4")
    ds = nc.Dataset(os.path.join(out, "SWI_TDRGRID_F16_D19980716.nc"))
    try:
        filled = np.asarray(ds["n_ta19v_asc"][:]) > 0
        values = np.unique(np.asarray(ds["ta19v_asc"][:])[filled])
        assert ds.satellite == "F16"
        assert values == pytest.approx([222.0])   # no F-13 pixels present
        assert "f13_cross.nc" not in ds.source_granules
    finally:
        ds.close()

    # Each satellite still gets its own day written.
    assert os.path.exists(os.path.join(out, "SWI_TDRGRID_F13_D19980715.nc"))


def test_an_overlapping_archive_does_not_double_count(tmp_path, monkeypatch):
    """The same granule reached through two archives must be binned once."""
    d15, d16 = _archives(tmp_path)
    spare = tmp_path / "spare" / "CSU_SSMI_TDRBASE_V01R04_F13_D19980716"
    spare.mkdir(parents=True)
    # A second archive for the same satellite and day, holding the same granule.
    write_granule(str(spare / LATER_NC), LATER_START, nscan=NSCAN, lat0=30.125)

    single = str(tmp_path / "single")
    doubled = str(tmp_path / "doubled")
    _run(monkeypatch, [d15, d16, "--out", single])
    _run(monkeypatch, [d15, d16, str(spare), "--out", doubled])

    n_single, _ = _read(os.path.join(single, "SWI_TDRGRID_F13_D19980716.nc"))
    n_doubled, sources = _read(os.path.join(doubled,
                                            "SWI_TDRGRID_F13_D19980716.nc"))
    assert n_doubled == n_single
    assert sources.split().count(LATER_NC) == 1


# --------------------------------------------------------------------------
# Inputs and rebuilding
# --------------------------------------------------------------------------

def _tars(tmp_path):
    """The two day directories repacked as tars, as CLASS delivers them."""
    d15, d16 = _archives(tmp_path)
    tars = tmp_path / "tars"
    tars.mkdir()
    for src, name in ((d15, DIR_15), (d16, DIR_16)):
        with tarfile.open(str(tars / f"{name}.tar"), "w") as tf:
            for f in sorted(os.listdir(src)):
                tf.add(os.path.join(src, f), arcname=f)
    return str(tars / f"{DIR_15}.tar"), str(tars / f"{DIR_16}.tar")


def test_tar_archives_are_accepted_and_the_workdir_is_removed(tmp_path,
                                                              monkeypatch):
    """Pin the removal of the real extraction directory, not the tars' folder."""
    t15, t16 = _tars(tmp_path)
    work = str(tmp_path / "workdir")
    os.makedirs(work)
    monkeypatch.setattr(drv.tempfile, "mkdtemp", lambda *a, **k: work)
    out = str(tmp_path / "out")

    assert _run(monkeypatch, [t15, t16, "--out", out]) == 0

    # Extraction really happened, so the day carries the crossing granule.
    carried = scans_in_day(CROSS_START, NSCAN, "19980716")
    n16, sources = _read(os.path.join(out, "SWI_TDRGRID_F13_D19980716.nc"))
    assert n16 == (carried + NSCAN) * NPIX_LO
    assert CROSS_NC in sources
    # And the directory it unpacked into is gone.
    assert not os.path.exists(work)


def test_existing_output_is_kept_unless_forced(tmp_path, monkeypatch, capsys):
    """The forced run must actually rebuild, so change the input and look."""
    d15, d16 = _archives(tmp_path)
    out = str(tmp_path / "out")
    target = os.path.join(out, "SWI_TDRGRID_F13_D19980716.nc")

    _run(monkeypatch, [d15, d16, "--out", out])
    capsys.readouterr()

    def temperatures():
        nc = pytest.importorskip("netCDF4")
        ds = nc.Dataset(target)
        try:
            filled = np.asarray(ds["n_ta19v_asc"][:]) > 0
            return set(np.unique(np.asarray(ds["ta19v_asc"][:])[filled]))
        finally:
            ds.close()

    before = temperatures()
    assert 250.0 not in before

    # Rewrite the source with a distinct temperature the old file cannot hold.
    write_granule(os.path.join(d16, LATER_NC), LATER_START, nscan=NSCAN,
                  lat0=30.125,
                  values={"ta19v": np.full((NSCAN, NPIX_LO), 250.0)})

    _run(monkeypatch, [d15, d16, "--out", out])
    assert "already gridded" in capsys.readouterr().out
    assert temperatures() == before          # untouched without --force

    _run(monkeypatch, [d15, d16, "--out", out, "--force"])
    assert "already gridded" not in capsys.readouterr().out
    assert 250.0 in temperatures()           # genuinely rebuilt


def test_the_cache_releases_archives_a_sparse_run_cannot_reach(tmp_path,
                                                               monkeypatch):
    """Releasing only the archive one step back never fires on sparse dates."""
    t15, _ = _tars(tmp_path)
    calls = []
    real = drv.extract

    def counting(source, workdir):
        calls.append(source)
        return real(source, workdir)

    monkeypatch.setattr(drv, "extract", counting)
    cache = drv.GranuleCache(str(tmp_path / "work"))

    cache.get(t15, ("F13", "19980715"))
    cache.get(t15, ("F13", "19980715"))
    assert calls == [t15]                    # extracted once, then cached

    cache.retain_from("F13", "19980714")     # still reachable, keep it
    cache.get(t15, ("F13", "19980715"))
    assert calls == [t15]

    cache.retain_from("F13", "19980801")     # a jump past it in a sparse run
    cache.get(t15, ("F13", "19980715"))
    assert calls == [t15, t15]               # released, so extracted again


def test_the_cache_drops_another_satellites_archives(tmp_path, monkeypatch):
    t15, _ = _tars(tmp_path)
    calls = []
    real = drv.extract
    monkeypatch.setattr(drv, "extract",
                        lambda s, w: (calls.append(s), real(s, w))[1])
    cache = drv.GranuleCache(str(tmp_path / "work"))

    cache.get(t15, ("F13", "19980715"))
    cache.retain_from("F16", "19980715")     # the run moved to another sensor
    cache.get(t15, ("F13", "19980715"))
    assert calls == [t15, t15]


def test_a_refused_swath_is_reported_and_recorded(tmp_path, monkeypatch,
                                                  capsys):
    """An incomplete day must be visible in the run and in the file."""
    nc = pytest.importorskip("netCDF4")
    d16 = tmp_path / DIR_16
    d16.mkdir()
    write_granule(str(d16 / LATER_NC), LATER_START, nscan=NSCAN, lat0=30.125)
    write_granule(str(d16 / "broken.nc"),
                  _dt.datetime(1998, 7, 16, 6, 0, 0), nscan=NSCAN,
                  lat0=50.125, hires_scans=2 * NSCAN - 3)
    out = str(tmp_path / "out")

    _run(monkeypatch, [str(d16), "--out", out])
    printed = capsys.readouterr().out
    assert "INCOMPLETE" in printed
    assert "broken.nc" in printed
    assert "marked incomplete" in printed

    ds = nc.Dataset(os.path.join(out, "SWI_TDRGRID_F13_D19980716.nc"))
    try:
        assert ds.complete == "false"
        assert "broken.nc" in ds.rejected_granules
    finally:
        ds.close()


def test_an_empty_target_archive_does_not_yield_a_complete_day(tmp_path,
                                                               monkeypatch,
                                                               capsys):
    """The case a prior review demonstrated.

    With the previous day present and the target day's own archive empty, the
    driver builds the day from the crossing tail alone. That is a few minutes of
    data, and it must not be presented as a whole day.
    """
    nc = pytest.importorskip("netCDF4")
    d15 = tmp_path / DIR_15
    d16 = tmp_path / DIR_16
    d15.mkdir()
    d16.mkdir()                                   # deliberately empty
    write_granule(str(d15 / CROSS_NC), CROSS_START, nscan=NSCAN)
    out = str(tmp_path / "out")

    _run(monkeypatch, [str(d15), str(d16), "--out", out])
    printed = capsys.readouterr().out
    assert "INCOMPLETE" in printed

    ds = nc.Dataset(os.path.join(out, "SWI_TDRGRID_F13_D19980716.nc"))
    try:
        assert ds.complete == "false"
        assert ds.coverage_fraction < 0.01        # minutes, not a day
        assert ds.coverage_gaps_utc != ""
    finally:
        ds.close()


def test_a_damaged_archive_costs_its_own_day_only(tmp_path, monkeypatch,
                                                  capsys):
    """One truncated tar must not end the run for every other date."""
    t15, t16 = _tars(tmp_path)
    # Truncate the 16 July archive part way through.
    with open(t16, "r+b") as fh:
        fh.truncate(4096)
    out = str(tmp_path / "out")

    assert _run(monkeypatch, [t15, t16, "--out", out]) == 0
    printed = capsys.readouterr().out

    # The 15th still builds from its own intact archive.
    assert os.path.exists(os.path.join(out, "SWI_TDRGRID_F13_D19980715.nc"))
    assert "unreadable" in printed or "no granules" in printed


def test_inputs_without_a_date_in_the_name_are_skipped(tmp_path, monkeypatch,
                                                       capsys):
    junk = tmp_path / "unnamed"
    junk.mkdir()
    assert _run(monkeypatch, [str(junk), "--out", str(tmp_path / "out")]) == 1
    assert "no usable inputs" in capsys.readouterr().out
