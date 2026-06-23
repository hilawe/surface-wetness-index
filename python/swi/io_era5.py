"""Load ERA5-Land monthly soil moisture, normalized to our grid orientation.

Returns latitude ascending (south first) and longitude on 0..360 ascending, to
match the CSU product grid. Handles the new CDS delivery quirk where a zip
containing the NetCDF is returned even for data_format netcdf.
"""

import os

import numpy as np

DATASET = "reanalysis-era5-land-monthly-means"
VARIABLE = "volumetric_soil_water_layer_1"


def load_swvl1(month, path):
    """Return (lat_asc, lon_0360, sm) for ERA5-Land swvl1 for 'YYYYMM'.

    Uses the local file at `path` if present, else downloads via the CDS API.
    """
    import netCDF4 as nc

    if not os.path.exists(path):
        import cdsapi
        cdsapi.Client().retrieve(DATASET, {
            "product_type": "monthly_averaged_reanalysis",
            "variable": VARIABLE,
            "year": month[:4], "month": month[4:6], "time": "00:00",
            "data_format": "netcdf",
        }, path)

    with open(path, "rb") as fh:
        is_zip = fh.read(2) == b"PK"
    open_path = path
    if is_zip:
        import zipfile
        with zipfile.ZipFile(path) as z:
            inner = next(n for n in z.namelist() if n.endswith(".nc"))
            z.extract(inner, os.path.dirname(path) or ".")
            open_path = os.path.join(os.path.dirname(path) or ".", inner)

    ds = nc.Dataset(open_path)
    try:
        lat = np.asarray(ds["latitude"][:] if "latitude" in ds.variables
                         else ds["lat"][:], dtype=np.float64)
        lon = np.asarray(ds["longitude"][:] if "longitude" in ds.variables
                         else ds["lon"][:], dtype=np.float64)
        vname = next((v for v in ds.variables
                      if v.lower() == "swvl1"
                      or "soil_water" in getattr(ds[v], "long_name", "").lower()),
                     None)
        if vname is None:
            raise ValueError(f"no soil-water variable in {open_path}")
        arr = np.ma.filled(ds[vname][:], np.nan).astype(np.float64)
    finally:
        ds.close()

    arr = np.squeeze(arr)
    if lat[0] > lat[-1]:
        lat = lat[::-1]; arr = arr[::-1, :]
    lon = np.where(lon < 0, lon + 360.0, lon)
    order = np.argsort(lon)
    return lat, lon[order], arr[:, order]
