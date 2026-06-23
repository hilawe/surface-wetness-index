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
