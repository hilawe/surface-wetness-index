"""ESA CCI soil-moisture loader: URL form and monthly compositing/orientation.

The compositing test writes tiny synthetic CCI-format files (no network) and
checks that load_cci_monthly drops flagged cells, averages clean days, and
returns our grid orientation (latitude ascending, longitude 0..360).
"""

import numpy as np
import pytest

from swi import io_esacci as io


def test_cci_url_and_filename():
    fn = io.cci_filename("20230701")
    assert fn == "ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-20230701000000-fv09.2.nc"
    url = io.cci_url("20230701")
    assert url.endswith("/COMBINED/v09.2/2023/" + fn)
    assert url.startswith("https://data.cci.ceda.ac.uk/thredds/fileServer/")


def _write_day(path, date, sm, flag):
    nc = pytest.importorskip("netCDF4")
    lat = np.array([45.0, 15.0, -15.0, -45.0])      # north-first (CCI native)
    lon = np.array([-135.0, -45.0, 45.0, 135.0])    # -180..180
    ds = nc.Dataset(path, "w")
    ds.createDimension("time", 1)
    ds.createDimension("lat", 4)
    ds.createDimension("lon", 4)
    ds.createVariable("lat", "f8", ("lat",))[:] = lat
    ds.createVariable("lon", "f8", ("lon",))[:] = lon
    v = ds.createVariable("sm", "f4", ("time", "lat", "lon"), fill_value=-9999.0)
    v[0] = sm
    f = ds.createVariable("flag", "i2", ("time", "lat", "lon"), fill_value=-9999)
    f[0] = flag
    ds.close()


def test_load_cci_monthly_composites_and_reorients(tmp_path):
    pytest.importorskip("netCDF4")
    native = np.array([[c + r * 10 for c in range(4)] for r in range(4)], float)
    flag0 = np.zeros((4, 4), np.int16)
    flag0[1, 1] = 1                                 # flag one cell on both days

    d1 = io.cci_filename("20230701")
    d2 = io.cci_filename("20230702")
    _write_day(tmp_path / d1, "20230701", native, flag0)
    _write_day(tmp_path / d2, "20230702", native + 2.0, flag0)

    lat, lon, mean, ndays = io.load_cci_monthly(
        "202307", str(tmp_path), download=False)

    # orientation: latitude ascending (south first), longitude 0..360 ascending
    np.testing.assert_allclose(lat, [-45, -15, 15, 45])
    np.testing.assert_allclose(lon, [45, 135, 225, 315])
    assert np.all(np.diff(lon) > 0)

    # native cell (row 0 lat=45, col 2 lon=45): mean of 2 and 4 -> 3 over 2 days,
    # lands at row 3 (lat 45) col 0 (lon 45) after reorientation
    assert mean[3, 0] == pytest.approx(3.0)
    assert ndays[3, 0] == 2

    # flagged cell (row 1 lat=15, col 1 lon=-45 -> 315) -> NaN, zero valid days,
    # at row 2 (lat 15) col 3 (lon 315)
    assert np.isnan(mean[2, 3])
    assert ndays[2, 3] == 0
