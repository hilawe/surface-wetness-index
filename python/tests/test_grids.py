"""Grid specs and swath binning, and that resolution is a free engine choice."""

import numpy as np
import pytest

from swi import core_numpy
from swi.grids import THIRD_DEGREE, QUARTER_DEGREE, GridSpec, bin_swath_to_grid


def test_grid_dimensions():
    assert THIRD_DEGREE.shape == (540, 1080)
    assert QUARTER_DEGREE.shape == (720, 1440)


def test_cell_index_corners_and_center():
    g = QUARTER_DEGREE
    # south-west corner
    assert g.cell_index(-90.0, -180.0) == (0, 0)
    # just north and east of the SW corner -> next cells
    assert g.cell_index(-89.9, -179.9) == (0, 0)
    assert g.cell_index(-89.74, -179.74) == (1, 1)
    # longitude wrap: +180 wraps to the -180 origin column
    ilat, ilon = g.cell_index(0.0, 180.0)
    assert ilon == 0


def test_bin_swath_simple_mean():
    g = QUARTER_DEGREE
    # three pixels in the same cell with different 19V values -> mean
    lat = np.array([0.10, 0.11, 0.12])
    lon = np.array([0.10, 0.11, 0.12])
    tb = np.tile(np.array([250., 240., 260., 255., 250., 260., 255.]), (3, 1))
    tb[:, 0] = [270., 280., 290.]            # 19V varies
    grid = bin_swath_to_grid(g, lat, lon, tb)
    ilat, ilon = g.cell_index(lat[0], lon[0])
    assert grid[ilat, ilon, 0] == pytest.approx(280.0)   # mean of 270,280,290
    # an untouched cell is fill (NaN)
    assert np.isnan(grid[0, 0, 0])


def test_resolution_is_a_free_choice():
    """A uniform Tb field yields identical engine output at 1/3 and 1/4 deg."""
    rng = np.random.default_rng(0)
    n = 20000
    lat = rng.uniform(-89, 89, n)
    lon = rng.uniform(-179, 179, n)
    tb = np.tile(np.array([272., 258., 274., 268., 256., 266., 259.]), (n, 1))

    for g in (THIRD_DEGREE, QUARTER_DEGREE):
        grid = bin_swath_to_grid(g, lat, lon, tb)
        populated = np.isfinite(grid).all(axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            res = core_numpy.evaluate_kelvin(grid[populated])
        # uniform input -> every populated cell gives the same result
        assert np.unique(res.snow).size == 1
        assert np.unique(res.temp).size == 1


def test_check_grid_enforces_the_fixed_grid_invariant():
    """The workers cache cell masks and the atlas from the first day and reuse
    them all month; check_grid makes that invariant fail loudly if a later day
    arrives on a different grid, rather than silently pairing it with stale
    cached arrays (round-three review, minor 8)."""
    from swi import grids
    lat = np.linspace(-89.875, 89.875, 720)
    lon = np.linspace(0.125, 359.875, 1440)
    # First day: returns the reference grid to cache.
    lat0, lon0 = grids.check_grid(lat, lon, None, None)
    assert np.array_equal(lat0, lat) and np.array_equal(lon0, lon)
    # Same grid on a later day: accepted, reference unchanged.
    assert grids.check_grid(lat, lon, lat0, lon0)[0] is lat0
    # A reordered longitude vector (same shape) must be rejected, not accepted.
    with pytest.raises(ValueError, match="grid mismatch"):
        grids.check_grid(lat, lon[::-1], lat0, lon0)
    # A different latitude grid must be rejected too.
    with pytest.raises(ValueError, match="grid mismatch"):
        grids.check_grid(np.linspace(-89.5, 89.5, 720), lon, lat0, lon0)
