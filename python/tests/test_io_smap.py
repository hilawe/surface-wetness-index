"""SMAP loader: grid separability, quality screening, and area-binning.

All tests write tiny synthetic SPL3SMP_E-shaped files, so nothing here touches
the network or a NASA Earthdata login. The binning test is the important one,
because going from the 9 km grid to 0.25 degree is a coarsening and a wrong
index calculation there would silently misplace every retrieval.
"""

import numpy as np
import pytest

from swi import io_smap as io


def _write(path, group_vars):
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as f:
        g = f.create_group(io.GROUP_AM)
        for k, v in group_vars.items():
            g.create_dataset(k, data=v)


def test_load_coords_collapses_separable_grid(tmp_path):
    pytest.importorskip("h5py")
    lat1d = np.array([30.0, 10.0, -10.0])          # north-first, as delivered
    lon1d = np.array([-90.0, 0.0, 90.0, 180.0])
    lat2 = np.repeat(lat1d[:, None], 4, axis=1)
    lon2 = np.repeat(lon1d[None, :], 3, axis=0)
    _write(tmp_path / io.COORDS_FILE, {"latitude": lat2, "longitude": lon2})

    la, lo = io.load_coords(str(tmp_path))
    np.testing.assert_allclose(la, lat1d)
    np.testing.assert_allclose(lo, lon1d)


def test_load_coords_rejects_a_non_separable_grid(tmp_path):
    """A grid whose rows are not constant latitude must fail loudly.

    The binning below assumes separability. If that assumption were ever false
    and went unchecked, every retrieval would be placed in the wrong cell and
    the comparison would still produce a plausible-looking number.
    """
    pytest.importorskip("h5py")
    lat2 = np.array([[30.0, 31.0, 32.0, 33.0],     # latitude varies along the row
                     [10.0, 10.0, 10.0, 10.0],
                     [-10.0, -10.0, -10.0, -10.0]])
    lon2 = np.repeat(np.array([-90.0, 0.0, 90.0, 180.0])[None, :], 3, axis=0)
    _write(tmp_path / io.COORDS_FILE, {"latitude": lat2, "longitude": lon2})

    with pytest.raises(ValueError, match="not constant along rows"):
        io.load_coords(str(tmp_path))


def test_read_day_applies_fill_and_quality_bit(tmp_path):
    pytest.importorskip("h5py")
    sm = np.array([[0.10, 0.20], [0.30, -9999.0]])
    qf = np.array([[0, 1], [0, 0]], dtype=np.uint16)   # bit0=1 -> not recommended
    p = tmp_path / "smap_am_20230701.nc4"
    _write(p, {"soil_moisture": sm, "retrieval_qual_flag": qf})

    screened = io._read_day(str(p), screen="recommended")
    assert screened[0, 0] == pytest.approx(0.10)
    assert np.isnan(screened[0, 1])        # flagged not-recommended
    assert screened[1, 0] == pytest.approx(0.30)
    assert np.isnan(screened[1, 1])        # fill value

    unscreened = io._read_day(str(p), screen="none")
    assert unscreened[0, 1] == pytest.approx(0.20)   # flag ignored
    assert np.isnan(unscreened[1, 1])                # fill still dropped


def test_bin_to_target_averages_every_fine_cell_in_a_coarse_cell():
    """Coarsening must AVERAGE the fine cells, and assign them to the NEAREST
    target cell rather than the one below.

    The offsets are chosen so rounding-to-nearest and rounding-down disagree.
    A fine center 0.6 of a cell along belongs to target index 1, not 0. Rounding
    down instead would shift every such retrieval one cell south and west, a
    systematic half-cell offset that still produces a plausible-looking map, so
    the test has to be able to see it.
    """
    field = np.arange(9, dtype=float).reshape(3, 3)
    slat = np.array([0.0, 0.6, 1.0])
    slon = np.array([0.0, 54.0, 90.0])     # 54 is 0.6 of a 90 degree cell
    tlat = np.array([0.0, 1.0, 2.0])
    tlon = np.array([0.0, 90.0, 180.0, 270.0])

    ssum, scnt = io._bin_to_target(field, slat, slon, tlat, tlon)
    # fine rows 1 and 2 both land on target row 1, likewise the columns
    assert scnt[0, 0] == 1 and ssum[0, 0] == pytest.approx(0)
    assert scnt[0, 1] == 2 and ssum[0, 1] == pytest.approx(1 + 2)
    assert scnt[1, 0] == 2 and ssum[1, 0] == pytest.approx(3 + 6)
    assert scnt[1, 1] == 4 and ssum[1, 1] == pytest.approx(4 + 5 + 7 + 8)
    assert scnt[2, 2] == 0


def test_bin_to_target_skips_nan_and_wraps_longitude():
    field = np.array([[np.nan, 1.0]])
    slat = np.array([0.0])
    slon = np.array([0.0, -90.0])          # -90 is 270 east, past the seam
    tlat = np.array([0.0, 1.0])
    tlon = np.array([0.0, 90.0, 180.0, 270.0])

    ssum, scnt = io._bin_to_target(field, slat, slon, tlat, tlon)
    assert scnt.sum() == 1                  # the NaN contributed nothing
    assert ssum.sum() == pytest.approx(1.0)
    assert ssum[0, 3] == pytest.approx(1.0)  # wrapped to the 270 degree column


def test_load_smap_monthly_composites_daily_means(tmp_path):
    """Monthly value is the mean of the DAILY target-grid means.

    Day 1 and day 2 differ, so a cell present on both days must land halfway.
    A cell present on only one day must equal that day and report n_days = 1.
    """
    pytest.importorskip("h5py")
    lat1d = np.array([0.0, 1.0])
    lon1d = np.array([0.0, 180.0])         # global span, as the product grid is
    _write(tmp_path / io.COORDS_FILE,
           {"latitude": np.repeat(lat1d[:, None], 2, axis=1),
            "longitude": np.repeat(lon1d[None, :], 2, axis=0)})

    qf = np.zeros((2, 2), dtype=np.uint16)
    _write(tmp_path / "smap_am_20230701.nc4",
           {"soil_moisture": np.array([[0.10, 0.20], [0.30, 0.40]]),
            "retrieval_qual_flag": qf})
    _write(tmp_path / "smap_am_20230702.nc4",
           {"soil_moisture": np.array([[0.30, -9999.0], [0.30, 0.40]]),
            "retrieval_qual_flag": qf})

    mean, nday, stats = io.load_smap_monthly(
        "202307", str(tmp_path), lat1d, lon1d, download=False, verbose=False)

    assert mean[0, 0] == pytest.approx(0.20)   # (0.10 + 0.30) / 2
    assert nday[0, 0] == 2
    assert mean[0, 1] == pytest.approx(0.20)   # only day 1 had a value
    assert nday[0, 1] == 1
    assert stats["n_days_used"] == 2
    assert stats["fine_successful_retrievals"] == 7
    assert stats["fine_kept_by_screen"] == 7


# --- behaviors added after a mutation run showed the suite did not bind them ---

class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=None):
        yield self._body


class _FakeSession:
    """Duck-typed stand-in for requests.Session, so no network is touched."""

    def __init__(self, body):
        self._body = body
        self.url = None

    def get(self, url, **kw):
        self.url = url
        return _FakeResponse(self._body)


def test_opendap_subset_refuses_a_body_that_is_not_hdf5(tmp_path):
    """A Hyrax error page arrives with HTTP 200, so status is not evidence.

    Without this guard a failed request would be written to the cache path and
    every later run would read the error page as if it were a granule.
    """
    out = tmp_path / "day.nc4"
    sess = _FakeSession(b"<!DOCTYPE html><html>Hyrax - Bad Request (400)</html>")
    with pytest.raises(ValueError, match="not netCDF4/HDF5"):
        io._opendap_subset(sess, "GRANULE_UR", ["soil_moisture"], str(out))
    assert not out.exists()
    assert not (tmp_path / "day.nc4.part").exists()   # the partial is cleaned up


def test_opendap_subset_accepts_a_real_subset_and_builds_the_request(tmp_path):
    """The eight-byte signature is not enough, so the fixture is a real file.

    An earlier version passed a 24-byte stub carrying only the magic number, and
    a review pointed out that a cleanly truncated response would look exactly
    like that and be promoted into the cache.
    """
    pytest.importorskip("h5py")
    good = tmp_path / "good.nc4"
    _write(good, {"soil_moisture": np.zeros(io.GRID_SHAPE, np.float32),
                  "retrieval_qual_flag": np.zeros(io.GRID_SHAPE, np.uint16)})
    body = open(good, "rb").read()

    out = tmp_path / "day.nc4"
    sess = _FakeSession(body)
    n = io._opendap_subset(sess, "GRANULE_UR", ["soil_moisture", "retrieval_qual_flag"],
                           str(out))
    assert out.exists() and n == len(body)
    # the constrained-subset form is what keeps a day at ~4 MB instead of ~629 MB
    assert ".dap.nc4?dap4.ce=" in sess.url
    assert f"/{io.GROUP_AM}/soil_moisture" in sess.url
    assert "%3B" in sess.url                          # encoded separator, not a raw ;


def test_bin_to_target_drops_fine_rows_outside_the_target_latitudes():
    """EASE-Grid stops near 84.7 degrees while the product grid runs to 90.

    Fine cells whose row index falls outside the target must be discarded. If
    they are not, they land at an out-of-range flat index and corrupt the grid.
    """
    field = np.array([[1.0, 2.0], [100.0, 200.0]])
    slat = np.array([0.0, 5.0])          # second row is far outside the target span
    slon = np.array([0.0, 90.0])
    tlat = np.array([0.0, 1.0])
    tlon = np.array([0.0, 90.0, 180.0, 270.0])

    ssum, scnt = io._bin_to_target(field, slat, slon, tlat, tlon)
    assert scnt.sum() == 2               # only the in-range row contributed
    assert ssum[0, 0] == pytest.approx(1.0)
    assert ssum[0, 1] == pytest.approx(2.0)
    assert ssum.sum() == pytest.approx(3.0)


def test_load_smap_monthly_reports_the_true_quality_retention(tmp_path):
    """The retention statistic is what tells a reader the screen was not free.

    It is reported in the paper, so a wrong value is a false claim even though
    every skill number would be unchanged.
    """
    pytest.importorskip("h5py")
    lat1d = np.array([0.0, 1.0])
    lon1d = np.array([0.0, 180.0])         # global span, as the product grid is
    _write(tmp_path / io.COORDS_FILE,
           {"latitude": np.repeat(lat1d[:, None], 2, axis=1),
            "longitude": np.repeat(lon1d[None, :], 2, axis=0)})
    # four valid retrievals, one of them flagged not-recommended
    _write(tmp_path / "smap_am_20230701.nc4",
           {"soil_moisture": np.array([[0.10, 0.20], [0.30, 0.40]]),
            "retrieval_qual_flag": np.array([[0, 1], [0, 0]], dtype=np.uint16)})

    _, _, stats = io.load_smap_monthly(
        "202307", str(tmp_path), lat1d, lon1d, download=False, verbose=False)

    assert stats["fine_successful_retrievals"] == 4
    assert stats["fine_kept_by_screen"] == 3
    assert stats["kept_fraction_of_successful"] == pytest.approx(0.75)


def test_load_coords_rejects_longitude_varying_down_a_column(tmp_path):
    """The longitude half of the separability check, which had no test.

    A mutation run showed the suite bound the latitude assertion and not this
    one, so a grid separable in latitude but not longitude would have passed.
    """
    pytest.importorskip("h5py")
    lat2 = np.repeat(np.array([30.0, 10.0, -10.0])[:, None], 4, axis=1)
    lon2 = np.array([[-90.0, 0.0, 90.0, 180.0],       # longitude varies down
                     [-89.0, 1.0, 91.0, 181.0],       # each column
                     [-88.0, 2.0, 92.0, 182.0]])
    _write(tmp_path / io.COORDS_FILE, {"latitude": lat2, "longitude": lon2})

    with pytest.raises(ValueError, match="not constant along columns"):
        io.load_coords(str(tmp_path))


def test_bin_to_target_refuses_a_longitude_grid_that_is_not_global():
    """The wrap-around modulo is only safe on a grid spanning the whole globe.

    On a regional grid it would fold far-away meridians into the domain instead
    of discarding them, and the map would look unremarkable. Latitude is
    protected by an explicit range test, longitude by this refusal.
    """
    field = np.array([[1.0, 2.0]])
    slat = np.array([0.0])
    slon = np.array([0.0, 1.0])
    tlat = np.array([0.0, 1.0])
    tlon = np.array([0.0, 1.0, 2.0])       # a 3 degree regional strip

    with pytest.raises(ValueError, match="not 360"):
        io._bin_to_target(field, slat, slon, tlat, tlon)


# --- behaviors added after an independent review found them unbound ---

def test_month_files_ignores_other_months(tmp_path):
    """Offline loading must not fold a neighboring month into the composite.

    The earlier code globbed every `smap_am_*` file. Downloading August into the
    same directory and rerunning July would have produced a July-August composite
    while the result file still said 202307, with nothing to reveal it.
    """
    for name in ("smap_am_20230701.nc4", "smap_am_20230731.nc4",
                 "smap_am_20230801.nc4", "smap_am_20230601.nc4",
                 "smap_coords_9km.nc4", "notes.txt"):
        (tmp_path / name).write_bytes(b"")

    got = io.month_files("202307", str(tmp_path))
    assert [d for d, _ in got] == ["20230701", "20230731"]


def test_load_smap_monthly_excludes_a_neighboring_month(tmp_path):
    """The month filter must hold end to end, not only in the helper."""
    pytest.importorskip("h5py")
    lat1d = np.array([0.0, 1.0])
    lon1d = np.array([0.0, 180.0])
    _write(tmp_path / io.COORDS_FILE,
           {"latitude": np.repeat(lat1d[:, None], 2, axis=1),
            "longitude": np.repeat(lon1d[None, :], 2, axis=0)})
    qf = np.zeros((2, 2), dtype=np.uint16)
    _write(tmp_path / "smap_am_20230701.nc4",
           {"soil_moisture": np.full((2, 2), 0.10), "retrieval_qual_flag": qf})
    _write(tmp_path / "smap_am_20230801.nc4",       # August, must be ignored
           {"soil_moisture": np.full((2, 2), 0.90), "retrieval_qual_flag": qf})

    mean, nday, stats = io.load_smap_monthly(
        "202307", str(tmp_path), lat1d, lon1d, download=False, verbose=False)

    assert stats["n_days_used"] == 1
    assert mean[0, 0] == pytest.approx(0.10)     # not 0.50, the two-month mean
    assert nday[0, 0] == 1


def test_read_day_screens_differ_on_an_unsuccessful_retrieval(tmp_path):
    """Bit 2 marks a retrieval the product itself calls unsuccessful.

    Ignoring every quality bit admits those values, which is why "none" is not a
    defensible sensitivity and "successful" exists as the relaxed screen.
    """
    pytest.importorskip("h5py")
    sm = np.array([[0.10, 0.20, 0.30]])
    #        recommended,  not recommended but successful,  unsuccessful (bit 2)
    qf = np.array([[0b000, 0b001, 0b101]], dtype=np.uint16)
    p = tmp_path / "smap_am_20230701.nc4"
    _write(p, {"soil_moisture": sm, "retrieval_qual_flag": qf})

    rec = io._read_day(str(p), screen="recommended")
    suc = io._read_day(str(p), screen="successful")
    none = io._read_day(str(p), screen="none")

    assert np.isfinite(rec).sum() == 1           # only the recommended one
    assert np.isfinite(suc).sum() == 2           # recommended plus uncertain
    assert np.isfinite(none).sum() == 3          # admits the unsuccessful one
    assert np.isnan(suc[0, 2])                   # the unsuccessful one is dropped


def test_read_day_rejects_an_unknown_screen(tmp_path):
    pytest.importorskip("h5py")
    p = tmp_path / "smap_am_20230701.nc4"
    _write(p, {"soil_moisture": np.zeros((1, 1)),
               "retrieval_qual_flag": np.zeros((1, 1), np.uint16)})
    with pytest.raises(ValueError, match="unknown screen"):
        io._read_day(str(p), screen="lenient")


def test_verify_subset_rejects_a_truncated_download(tmp_path):
    """A cleanly truncated response still carries the HDF5 signature.

    The guard therefore has to open the file and look for the variables that were
    requested, on the shape the grid actually has.
    """
    pytest.importorskip("h5py")
    short = tmp_path / "short.nc4"
    _write(short, {"soil_moisture": np.zeros((10, 10), np.float32)})

    with pytest.raises(ValueError, match="missing variable"):
        io.verify_subset(str(short), ["soil_moisture", "retrieval_qual_flag"])
    with pytest.raises(ValueError, match="has shape"):
        io.verify_subset(str(short), ["soil_moisture"])


def test_opendap_subset_deletes_a_download_that_fails_verification(tmp_path):
    pytest.importorskip("h5py")
    wrong = tmp_path / "wrong.nc4"
    _write(wrong, {"soil_moisture": np.zeros((10, 10), np.float32)})
    body = open(wrong, "rb").read()

    out = tmp_path / "day.nc4"
    sess = _FakeSession(body)
    with pytest.raises(ValueError):
        io._opendap_subset(sess, "GRANULE_UR", ["soil_moisture"], str(out))
    assert not out.exists()
    assert not (tmp_path / "day.nc4.part").exists()
