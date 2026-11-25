"""ERA5-Land soil-moisture loader: archive handling and grid orientation.

The Climate Data Store delivers a zip archive holding the NetCDF even when a
plain NetCDF is asked for, so the loader has to open both. The tests write
tiny synthetic files, with no network.
"""

import os
import zipfile

import numpy as np
import pytest

from swi import io_era5 as io


def _write_field(path):
    nc = pytest.importorskip("netCDF4")
    lat = np.array([45.0, 15.0, -15.0, -45.0])       # north first, as delivered
    lon = np.array([-135.0, -45.0, 45.0, 135.0])     # -180..180, as delivered
    field = np.arange(16, dtype="f4").reshape(4, 4) / 100.0
    ds = nc.Dataset(path, "w")
    ds.createDimension("valid_time", 1)
    ds.createDimension("latitude", 4)
    ds.createDimension("longitude", 4)
    ds.createVariable("latitude", "f8", ("latitude",))[:] = lat
    ds.createVariable("longitude", "f8", ("longitude",))[:] = lon
    v = ds.createVariable("swvl1", "f4", ("valid_time", "latitude", "longitude"))
    v[0] = field
    ds.close()
    return field


def _zipped(tmp_path):
    inner = tmp_path / "build" / "data_stream-moda.nc"
    inner.parent.mkdir()
    field = _write_field(str(inner))
    store = tmp_path / "store"
    store.mkdir()
    archive = store / "era5land_swvl1_202307.nc"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(inner, "data_stream-moda.nc")
    return store, archive, field


def test_a_plain_file_is_read_in_the_product_orientation(tmp_path):
    path = tmp_path / "era5land_swvl1_202307.nc"
    field = _write_field(str(path))
    lat, lon, sm = io.load_swvl1("202307", str(path))
    assert list(lat) == [-45.0, -15.0, 15.0, 45.0]
    assert list(lon) == [45.0, 135.0, 225.0, 315.0]
    # the south row comes first, and the eastern half leads the western
    assert sm[0, 0] == pytest.approx(field[3, 2])
    assert sm[3, 3] == pytest.approx(field[0, 1])


def test_an_archive_reads_the_same_as_the_file_inside_it(tmp_path):
    plain = tmp_path / "plain.nc"
    _write_field(str(plain))
    _store, archive, _field = _zipped(tmp_path)
    want = io.load_swvl1("202307", str(plain))
    got = io.load_swvl1("202307", str(archive))
    for a, b in zip(want, got):
        assert np.array_equal(a, b)


def test_reading_an_archive_writes_nothing_beside_it(tmp_path):
    """The member used to be unpacked into the input data directory."""
    store, archive, _field = _zipped(tmp_path)
    before = sorted(os.listdir(store))
    io.load_swvl1("202307", str(archive))
    assert sorted(os.listdir(store)) == before == ["era5land_swvl1_202307.nc"]
