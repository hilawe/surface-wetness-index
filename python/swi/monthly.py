"""Monthly compositing of daily Surface Wetness Index retrievals.

Streams a month of daily grids through the engine and accumulates per-cell
monthly statistics without holding every day in memory. Composites are kept
separate for the ascending and descending passes (they sample different local
times). Only valid retrievals contribute: WET and RTEMP sentinels (-99) and gaps
(-100) are excluded from the means.

Monthly fields per pass:
- wetness_index_mean      : mean of daily WET where WET >= 0
- land_skin_temperature_mean : mean of daily RTEMP where RTEMP > -90
- snow_frequency          : fraction of observed days flagged snow (SNOW > 0)
- n_observations          : number of observed days (cell seen, not a gap)
- n_wet                   : number of days with a valid wetness retrieval
"""

import numpy as np


class Accumulator:
    """Per-cell streaming accumulator for one pass."""

    def __init__(self, shape):
        self.shape = shape
        self.wet_sum = np.zeros(shape, np.float64)
        self.wet_cnt = np.zeros(shape, np.int32)
        self.temp_sum = np.zeros(shape, np.float64)
        self.temp_cnt = np.zeros(shape, np.int32)
        self.snow_cnt = np.zeros(shape, np.int32)
        self.obs_cnt = np.zeros(shape, np.int32)

    def add(self, res):
        """Add one daily pass result (dict with 'wet','temp','snow')."""
        wet, temp, snow = res["wet"], res["temp"], res["snow"]
        wv = np.isfinite(wet) & (wet >= 0.0)
        self.wet_sum[wv] += wet[wv]
        self.wet_cnt += wv
        tv = np.isfinite(temp) & (temp > -90.0)
        self.temp_sum[tv] += temp[tv]
        self.temp_cnt += tv
        # observed = cell was seen (not no-data fill -128, not gap -100)
        obs = (snow != -128) & (snow != -100)
        self.obs_cnt += obs
        self.snow_cnt += (snow > 0)

    def result(self):
        wet_mean = np.where(self.wet_cnt > 0, self.wet_sum / np.maximum(self.wet_cnt, 1),
                            np.nan).astype(np.float32)
        temp_mean = np.where(self.temp_cnt > 0, self.temp_sum / np.maximum(self.temp_cnt, 1),
                             np.nan).astype(np.float32)
        snow_freq = np.where(self.obs_cnt > 0, self.snow_cnt / np.maximum(self.obs_cnt, 1),
                             np.nan).astype(np.float32)
        return {
            "wetness_index_mean": wet_mean,
            "land_skin_temperature_mean": temp_mean,
            "snow_frequency": snow_freq,
            "n_observations": self.obs_cnt.astype(np.int16),
            "n_wet": self.wet_cnt.astype(np.int16),
        }


def composite(files, engine, pass_list=("asc", "dsc")):
    """Composite a month of CSU daily files. Returns {pass: result dict}.

    engine : object with evaluate_kelvin (e.g. core_numpy or a calibrated wrapper).
    Files that fail to read or that lack required channels for a pass are skipped
    for that pass (and counted in n_skipped).
    """
    from . import io_csu_grid as io

    accs = {}
    lat = lon = None
    used = {p: 0 for p in pass_list}
    skipped = {p: 0 for p in pass_list}
    for f in files:
        for p in pass_list:
            try:
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = io.evaluate_file(f, pass_=p, engine=engine)
            except Exception:
                skipped[p] += 1
                continue
            if r.get("empty_channels"):
                skipped[p] += 1
                continue
            if lat is None:
                lat, lon = r["lat"], r["lon"]
            if p not in accs:
                accs[p] = Accumulator(r["wet"].shape)
            accs[p].add(r)
            used[p] += 1
    out = {p: accs[p].result() for p in accs}
    return {"lat": lat, "lon": lon, "by_pass": out, "used": used, "skipped": skipped}


class MeanAccumulator:
    """Generic NaN-aware per-cell streaming mean accumulator.

    A drop-in replacement for the ad-hoc ``sum / count`` patterns previously
    rewritten in several monthly-composite scripts. Lazy-allocated on the first
    ``add(field)`` so the caller does not need to know the shape ahead of time,
    and shape-agnostic on the trailing axes so it works for a 2-D scalar field
    or a (..., nchannel) per-channel stack.

    Usage:
        acc = MeanAccumulator()
        for field in stream:
            acc.add(field)
        mean = acc.mean()           # NaN where no observations contributed
        counts = acc.count()        # int array of contributions per cell
    """

    def __init__(self):
        self._sum = None
        self._count = None

    def add(self, field):
        """Add one field to the running mean. NaN cells do not contribute."""
        field = np.asarray(field, dtype=np.float64)
        if self._sum is None:
            self._sum = np.zeros_like(field)
            self._count = np.zeros(field.shape, dtype=np.int64)
        valid = np.isfinite(field)
        self._sum[valid] += field[valid]
        self._count[valid] += 1
        return self

    def mean(self):
        """Per-cell mean over the added fields. NaN where count is 0."""
        if self._sum is None:
            return None
        denom = np.where(self._count > 0, self._count, 1.0)
        return np.where(self._count > 0, self._sum / denom, np.nan)

    def count(self):
        """Per-cell contribution count."""
        if self._count is None:
            return None
        return self._count.copy()


def read_wet_product(path, pass_="dsc"):
    """Read a daily Surface Wetness Index NetCDF and return (wet, temp, snow).

    Centralizes the five-times-redefined product reader. ``pass_`` selects the
    ascending or descending pass. Returns float32 arrays; the fill value
    ``-999`` is propagated as-is (the consumer decides whether to mask it).
    """
    import netCDF4

    ds = netCDF4.Dataset(path)
    try:
        ds.set_auto_mask(False)
        suf = "_" + pass_
        wet = np.asarray(ds.variables["wetness_index" + suf][:], dtype=np.float32)
        temp = np.asarray(
            ds.variables["land_skin_temperature" + suf][:], dtype=np.float32)
        snow = np.asarray(ds.variables["snow_flag" + suf][:], dtype=np.int16)
    finally:
        ds.close()
    return wet, temp, snow
