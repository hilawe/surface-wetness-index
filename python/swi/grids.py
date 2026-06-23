"""Grid specifications and swath-to-grid binning.

The Basist engine is per-pixel and therefore resolution-agnostic: the grid is
purely an input-side choice. This module makes the choice explicit so switching
between 1/3 degree (the heritage/sibling resolution) and 1/4 degree (aligning
with the CSU brightness-temperature CDR and other modern CDRs) is a one-line
config change.

Convention (matches the sibling ssmis-hydro grids): global, south-first in
latitude, longitude fastest-varying, longitude origin at -180.
"""

from dataclasses import dataclass

import numpy as np

from .channels import N_CHANNELS


@dataclass(frozen=True)
class GridSpec:
    name: str
    dlat: float          # cell size in degrees (square cells)
    lon_origin: float = -180.0

    @property
    def nlon(self):
        return int(round(360.0 / self.dlat))

    @property
    def nlat(self):
        return int(round(180.0 / self.dlat))

    @property
    def shape(self):
        return (self.nlat, self.nlon)

    def cell_index(self, lat, lon):
        """Map lat/lon (degrees) to (ilat, ilon) integer cell indices.

        South-first latitude (ilat 0 at -90). Longitudes are wrapped into
        [lon_origin, lon_origin+360). Returns int arrays; out-of-range latitudes
        are clamped to the valid band.
        """
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        ilat = np.floor((lat + 90.0) / self.dlat).astype(np.int64)
        lon_wrapped = np.mod(lon - self.lon_origin, 360.0)
        ilon = np.floor(lon_wrapped / self.dlat).astype(np.int64)
        ilat = np.clip(ilat, 0, self.nlat - 1)
        ilon = np.clip(ilon, 0, self.nlon - 1)
        return ilat, ilon

    def flat_index(self, lat, lon):
        ilat, ilon = self.cell_index(lat, lon)
        return ilat * self.nlon + ilon


# Standard grids.
THIRD_DEGREE = GridSpec("third_degree", 1.0 / 3.0)     # 1080 x 540 (heritage)
QUARTER_DEGREE = GridSpec("quarter_degree", 0.25)      # 1440 x 720 (CSU CDR aligned)

GRIDS = {g.name: g for g in (THIRD_DEGREE, QUARTER_DEGREE)}


def bin_swath_to_grid(grid, lat, lon, tb, fill=np.nan):
    """Average swath pixels into grid cells.

    grid : GridSpec
    lat, lon : 1-D arrays of pixel coordinates (degrees), length n
    tb   : (n, 7) array of channel brightness temperatures (any units)
    fill : value for empty cells (default NaN)

    Returns (nlat, nlon, 7) array of per-cell channel means. Simple equal-weight
    binning; footprint-weighted resampling is a possible future refinement.
    """
    lat = np.asarray(lat, dtype=np.float64).ravel()
    lon = np.asarray(lon, dtype=np.float64).ravel()
    tb = np.asarray(tb, dtype=np.float64)
    if tb.shape[-1] != N_CHANNELS:
        raise ValueError(f"tb last axis must be {N_CHANNELS}, got {tb.shape}")
    tb = tb.reshape(-1, N_CHANNELS)
    if not (lat.shape[0] == lon.shape[0] == tb.shape[0]):
        raise ValueError("lat, lon, tb must share the pixel count")

    ncell = grid.nlat * grid.nlon
    flat = grid.flat_index(lat, lon)

    # Ignore pixels with non-finite Tb in any channel.
    valid = np.isfinite(tb).all(axis=1)
    flat_v = flat[valid]
    counts = np.bincount(flat_v, minlength=ncell).astype(np.float64)

    out = np.full((ncell, N_CHANNELS), fill, dtype=np.float64)
    nonempty = counts > 0
    for c in range(N_CHANNELS):
        sums = np.bincount(flat_v, weights=tb[valid, c], minlength=ncell)
        out[nonempty, c] = sums[nonempty] / counts[nonempty]

    return out.reshape(grid.nlat, grid.nlon, N_CHANNELS)
