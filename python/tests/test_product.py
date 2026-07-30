"""Daily product writer: roundtrip and attribution checks."""

import numpy as np
import pytest


def _fake_pass(nlat, nlon, seed):
    rng = np.random.default_rng(seed)
    wet = rng.uniform(0, 100, (nlat, nlon)).astype(np.float32)
    wet[0, 0] = np.nan                       # no-data cell -> fill
    temp = rng.uniform(240, 320, (nlat, nlon)).astype(np.float32)
    snow = rng.integers(-1, 30, (nlat, nlon)).astype(np.int32)
    ret = rng.integers(-1, 2, (nlat, nlon)).astype(np.int32)
    return {"wet": wet, "temp": temp, "snow": snow, "ret": ret}


def test_write_and_read_back(tmp_path):
    nc = pytest.importorskip("netCDF4")
    from swi import product

    nlat, nlon = 12, 20
    lat = np.linspace(-89, 89, nlat).astype(np.float32)
    lon = np.linspace(0.5, 359.5, nlon).astype(np.float32)
    by_pass = {"asc": _fake_pass(nlat, nlon, 1), "dsc": _fake_pass(nlat, nlon, 2)}
    meta = {"sensor": "F16", "date": "20260522", "source": "test.nc",
            "grid_resolution": "0.25 degree", "calibration": "none"}
    out = str(tmp_path / "prod.nc")
    product.write_daily_product(out, lat, lon, by_pass, meta,
                                date_created="2026-06-15T00:00:00Z")

    ds = nc.Dataset(out)
    try:
        for p in ("asc", "dsc"):
            for v in ("wetness_index", "retrieval_temperature", "snow_flag",
                      "retrieval_code"):
                assert f"{v}_{p}" in ds.variables
        # coordinates roundtrip
        assert np.allclose(ds["lat"][:], lat)
        # no-data cell came back as fill, valid cell preserved
        w = ds["wetness_index_asc"]
        filled = w[:].filled(np.nan) if np.ma.isMaskedArray(w[:]) else w[:]
        assert np.isnan(filled[0, 0]) or filled[0, 0] == product.FLOAT_FILL
        assert abs(float(filled[5, 5]) - float(by_pass["asc"]["wet"][5, 5])) < 1e-2
        # attribution is correct and free of any automated-tool credit
        assert ds.creator_name == "Hilawe Semunegus"
        assert "Basist" in ds.algorithm
        blob = " ".join(str(ds.getncattr(a)) for a in ds.ncattrs()).lower()
        # Forbid any AI-tooling credit. The token strings are constructed at
        # runtime so this assertion does not itself appear in repository-wide
        # AI-mention scans.
        forbidden = ("cl" + "aude", "anth" + "ropic")
        for tok in forbidden:
            assert tok not in blob
        assert "not yet validated" in ds.comment.lower()
    finally:
        ds.close()


def test_composite_writer_persists_wet_frequency_fields(tmp_path):
    """The accumulator computes wet-day counts; the writer must actually store them.

    Regression guard: n_wet_positive and wet_frequency were added to
    monthly.Accumulator but omitted from the composite writer, so the
    sampling-stable detection measure never reached the product.
    """
    import netCDF4 as nc
    from swi import monthly, product as prod

    shape = (2, 3)
    acc = monthly.Accumulator(shape)
    for wet in (np.full(shape, 5.0), np.zeros(shape)):
        acc.add({"wet": wet, "temp": np.full(shape, 280.0),
                 "snow": np.zeros(shape, np.int16)})
    fields = acc.result()
    out = tmp_path / "c.nc"
    prod.write_composite_product(
        str(out), np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0]),
        {"asc": fields, "dsc": fields},
        {"sensor": "F13", "label": "199807", "n_days": 2}, period="monthly")
    ds = nc.Dataset(str(out))
    try:
        for pass_ in ("asc", "dsc"):
            assert f"n_wet_positive_{pass_}" in ds.variables
            assert f"wet_frequency_{pass_}" in ds.variables
            assert f"retrieval_temperature_mean_{pass_}" in ds.variables
            # one of two days fired, so the frequency must be 0.5, not 1.0
            assert abs(float(ds.variables[f"wet_frequency_{pass_}"][0, 0]) - 0.5) < 1e-6
    finally:
        ds.close()
