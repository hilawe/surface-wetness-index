"""Write daily Surface Wetness Index products as CF-compliant NetCDF.

Emits the three retrievals (wetness index, land skin temperature, snow flag) for
ascending and descending passes on the input grid, with metadata following CF and
ACDD conventions. Attribution credits the Basist algorithm and NOAA NCEI; no AI
tooling appears in any author or creator field.

Fill convention: cells with no input observation are written as the fill value
(-999.0 for floats, -128 for integers). The algorithm's own sentinels are
retained as meaningful data: WET/RTEMP = -99 (retrieval attempted, unusable),
SNOW = -100 gap, -99 bad, -1 ice/glacial, 0 none, >0 scattering magnitude.
"""

import numpy as np

FLOAT_FILL = np.float32(-999.0)
INT_FILL = np.int16(-128)

REFERENCES = ("Basist et al. 1998 (J. Appl. Meteor., 37, 888-911); "
              "Grody 1991 (J. Geophys. Res., 96, 7423-7435); "
              "Grody and Basist 1996 (IEEE TGRS, 34, 237-249)")


def _f(a):
    """NaN -> float fill."""
    out = np.asarray(a, dtype=np.float32).copy()
    out[~np.isfinite(out)] = FLOAT_FILL
    return out


def write_daily_product(out_path, lat, lon, by_pass, meta, date_created=None):
    """Write a daily product file.

    lat, lon : 1-D coordinate arrays.
    by_pass  : dict {'asc': result, 'dsc': result}, each result a mapping with
               'temp','wet','snow','ret' 2-D arrays (nlat, nlon).
    meta     : dict with keys sensor, date (YYYYMMDD), source, grid_resolution,
               calibration (str describing the 85/91 handling).
    date_created : ISO string; if None, filled from the current UTC time.
    """
    import netCDF4 as nc

    if date_created is None:
        import datetime
        date_created = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")

    ds = nc.Dataset(out_path, "w", format="NETCDF4")
    try:
        ds.createDimension("lat", lat.size)
        ds.createDimension("lon", lon.size)

        vlat = ds.createVariable("lat", "f4", ("lat",))
        vlat.units = "degrees_north"; vlat.standard_name = "latitude"
        vlat.long_name = "latitude of grid cell center"
        vlat[:] = lat
        vlon = ds.createVariable("lon", "f4", ("lon",))
        vlon.units = "degrees_east"; vlon.standard_name = "longitude"
        vlon.long_name = "longitude of grid cell center"
        vlon[:] = lon

        specs = {
            "wetness_index": dict(dtype="f4", units="1", fill=FLOAT_FILL,
                                  long_name="Basist surface wetness index",
                                  valid_range=np.array([0.0, 100.0], "f4"),
                                  comment="0 dry to ~100; -99 = retrieval unusable"),
            "land_skin_temperature": dict(dtype="f4", units="K", fill=FLOAT_FILL,
                                          long_name="land skin (shelter-height) temperature",
                                          valid_range=np.array([150.0, 350.0], "f4"),
                                          comment="-99 = undefined"),
            "snow_flag": dict(dtype="i2", units="1", fill=INT_FILL,
                              long_name="snow/ice flag and scattering magnitude",
                              comment="0 none; -1 ice/glacial; -99 bad; -100 gap; "
                                      ">0 scattering magnitude"),
            "retrieval_code": dict(dtype="i2", units="1", fill=INT_FILL,
                                   long_name="decision-tree return code",
                                   comment="0 good; 1 water-condition; -1 rejected"),
        }
        src = {"wetness_index": "wet", "land_skin_temperature": "temp",
               "snow_flag": "snow", "retrieval_code": "ret"}

        for pass_ in ("asc", "dsc"):
            res = by_pass[pass_]
            for name, spec in specs.items():
                v = ds.createVariable(f"{name}_{pass_}", spec["dtype"],
                                      ("lat", "lon"), fill_value=spec["fill"],
                                      zlib=True, complevel=4)
                v.units = spec["units"]
                v.long_name = f"{spec['long_name']} ({pass_}ending pass)"
                v.coordinates = "lat lon"
                v.comment = spec["comment"]
                if "valid_range" in spec:
                    v.valid_range = spec["valid_range"]
                arr = res[src[name]]
                if spec["dtype"] == "f4":
                    v[:, :] = _f(arr)
                else:
                    a = np.asarray(arr, dtype=np.int16).copy()
                    a[a == -128] = INT_FILL
                    v[:, :] = a

        ds.Conventions = "CF-1.8 ACDD-1.3"
        ds.title = "Surface Wetness Index (Basist) daily product"
        ds.summary = ("Daily surface wetness index, land skin temperature, and "
                      "snow flag from the Basist signal-recognition decision tree "
                      "applied to CSU intercalibrated SSM/I(S) brightness "
                      "temperatures.")
        ds.institution = "DOC/NOAA/NCEI > National Centers for Environmental Information"
        ds.creator_name = "Hilawe Semunegus"
        ds.creator_institution = "NOAA NCEI"
        ds.algorithm = ("Basist signal-recognition decision tree (42 tests); "
                        "original author Alan Basist")
        ds.references = REFERENCES
        ds.source = meta.get("source", "")
        ds.platform = meta.get("sensor", "")
        ds.spatial_resolution = meta.get("grid_resolution", "0.25 degree")
        ds.processing_level = "Level 3 daily gridded"
        ds.calibration = meta.get("calibration", "none (91 GHz used as 85 GHz)")
        ds.comment = ("Research output, not yet validated. SSMIS 91 GHz substitutes "
                      "for the 85 GHz channels the coefficients were tuned on; see "
                      "the calibration attribute.")
        ds.date = meta.get("date", "")
        ds.date_created = date_created
        ds.geospatial_lat_min = float(lat.min())
        ds.geospatial_lat_max = float(lat.max())
        ds.geospatial_lon_min = float(lon.min())
        ds.geospatial_lon_max = float(lon.max())
    finally:
        ds.close()
    return out_path


def write_composite_product(out_path, lat, lon, by_pass, meta, period="monthly",
                            date_created=None):
    """Write a composite product. period is 'weekly' or 'monthly'.

    by_pass : dict {'asc': fields, 'dsc': fields}, each from
              monthly.Accumulator.result().
    meta    : dict with sensor, label (YYYYMM or YYYY-Wnn), n_days, source,
              grid_resolution, calibration, optional time_coverage_start/end.
    """
    import netCDF4 as nc

    if date_created is None:
        import datetime
        date_created = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")

    pw = period
    specs = {
        "wetness_index_mean": dict(dtype="f4", units="1", fill=FLOAT_FILL,
                                   long_name=f"{pw} mean Basist surface wetness index",
                                   valid_range=np.array([0.0, 100.0], "f4")),
        "land_skin_temperature_mean": dict(dtype="f4", units="K", fill=FLOAT_FILL,
                                           long_name=f"{pw} mean land skin temperature",
                                           valid_range=np.array([150.0, 350.0], "f4")),
        "snow_frequency": dict(dtype="f4", units="1", fill=FLOAT_FILL,
                               long_name="fraction of observed days flagged snow",
                               valid_range=np.array([0.0, 1.0], "f4")),
        "n_observations": dict(dtype="i2", units="1", fill=INT_FILL,
                               long_name="number of observed days"),
        "n_wet": dict(dtype="i2", units="1", fill=INT_FILL,
                      long_name="number of days with a valid wetness retrieval"),
    }

    ds = nc.Dataset(out_path, "w", format="NETCDF4")
    try:
        ds.createDimension("lat", lat.size)
        ds.createDimension("lon", lon.size)
        vlat = ds.createVariable("lat", "f4", ("lat",))
        vlat.units = "degrees_north"; vlat.standard_name = "latitude"; vlat[:] = lat
        vlon = ds.createVariable("lon", "f4", ("lon",))
        vlon.units = "degrees_east"; vlon.standard_name = "longitude"; vlon[:] = lon

        for pass_ in ("asc", "dsc"):
            if pass_ not in by_pass:
                continue
            fields = by_pass[pass_]
            for name, spec in specs.items():
                v = ds.createVariable(f"{name}_{pass_}", spec["dtype"],
                                      ("lat", "lon"), fill_value=spec["fill"],
                                      zlib=True, complevel=4)
                v.units = spec["units"]
                v.long_name = f"{spec['long_name']} ({pass_}ending pass)"
                v.coordinates = "lat lon"
                if "valid_range" in spec:
                    v.valid_range = spec["valid_range"]
                arr = fields[name]
                if spec["dtype"] == "f4":
                    v[:, :] = _f(arr)
                else:
                    a = np.asarray(arr, dtype=np.int16).copy()
                    a[a == -128] = INT_FILL
                    v[:, :] = a

        ds.Conventions = "CF-1.8 ACDD-1.3"
        ds.title = f"Surface Wetness Index (Basist) {pw} product"
        ds.summary = (f"{pw.capitalize()} mean surface wetness index, land skin "
                      "temperature, and snow frequency from the Basist signal-"
                      "recognition decision tree applied to CSU SSM/I(S) brightness "
                      "temperatures.")
        ds.institution = "DOC/NOAA/NCEI > National Centers for Environmental Information"
        ds.creator_name = "Hilawe Semunegus"
        ds.creator_institution = "NOAA NCEI"
        ds.algorithm = ("Basist signal-recognition decision tree (42 tests); "
                        "original author Alan Basist")
        ds.references = REFERENCES
        ds.source = meta.get("source", "")
        ds.platform = meta.get("sensor", "")
        ds.spatial_resolution = meta.get("grid_resolution", "0.25 degree")
        ds.processing_level = f"Level 3 {pw} gridded"
        ds.calibration = meta.get("calibration", "none (91 GHz used as 85 GHz)")
        ds.comment = ("Research output, not yet validated. Composited from daily "
                      "retrievals; only valid daily values contribute to the means.")
        label = meta.get("label", meta.get("month", meta.get("week", "")))
        if period == "monthly":
            ds.month = label
        else:
            ds.week = label
        if meta.get("time_coverage_start"):
            ds.time_coverage_start = meta["time_coverage_start"]
        if meta.get("time_coverage_end"):
            ds.time_coverage_end = meta["time_coverage_end"]
        ds.n_days = int(meta.get("n_days", 0))
        ds.date_created = date_created
        ds.geospatial_lat_min = float(lat.min())
        ds.geospatial_lat_max = float(lat.max())
        ds.geospatial_lon_min = float(lon.min())
        ds.geospatial_lon_max = float(lon.max())
    finally:
        ds.close()
    return out_path


def write_monthly_product(out_path, lat, lon, by_pass, meta, date_created=None):
    """Monthly composite (wrapper over write_composite_product)."""
    if "label" not in meta and "month" in meta:
        meta = {**meta, "label": meta["month"]}
    return write_composite_product(out_path, lat, lon, by_pass, meta,
                                   period="monthly", date_created=date_created)


def write_weekly_product(out_path, lat, lon, by_pass, meta, date_created=None):
    """Weekly composite (wrapper over write_composite_product)."""
    if "label" not in meta and "week" in meta:
        meta = {**meta, "label": meta["week"]}
    return write_composite_product(out_path, lat, lon, by_pass, meta,
                                   period="weekly", date_created=date_created)
