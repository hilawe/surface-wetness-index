"""Grid CSU SSM/I TDR BASE antenna temperatures onto the CSU 0.25 degree grid.

    python -m scripts.grid_tdr_day TAR_OR_DIR [TAR_OR_DIR ...] --out DIR

Each input is either a daily tar of orbit granules as delivered by NOAA CLASS,
or a directory of already-extracted granules. One gridded NetCDF is written per
day, named for the satellite and date, in the layout described in
`swi/io_tdr_swath.py`. Tars are extracted to a temporary directory and removed
afterwards, so no unpacked copy of the archive is left behind.

A day is built from the granules that fall in it by scan time, not from the
archive that carries its date. CLASS files a granule under the day it starts, so
the orbit that crosses midnight into a day sits in the previous day's archive.
Every daily archive inspected for 1998 ends with such a granule, and the window
it holds runs from 1 to 100 minutes depending on orbit phase. Each day is
therefore assembled from its own archive and the preceding one, with
`swi.io_tdr_swath` selecting the scans that belong. The following archive is
offered too, which under the start-day filing rule contributes nothing, and is
read only as a guard against an archive that files a granule by its end date.
The first day of a supplied run has no predecessor and is short by that day's
opening window, so supply one archive before the range wanted. The run reports
which days were built that way.

Archives are tracked by satellite as well as by date, so several satellites can
be passed in one invocation without one satellite's orbits reaching into
another's grid. A granule offered through more than one archive is binned once.

A day whose granules were not all usable is written with its refusals recorded
in the file and printed here, so an incomplete day is never mistaken for a whole
one.

Options:

    --out DIR    where to write the gridded days (default ../data/f13_1998_tdr_grid)
    --force      regrid days whose output already exists
"""

import datetime as _dt
import glob
import os
import re
import shutil
import sys
import tarfile
import tempfile

from swi import io_tdr_swath as tdr


def opt(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def day_key(path):
    """Satellite and date from a granule or tar name, or None."""
    m = re.search(r"_(F\d\d)_D(\d{8})", os.path.basename(path))
    return (m.group(1), m.group(2)) if m else None


def shift_day(date, days):
    """The date `days` away from a YYYYMMDD string, as YYYYMMDD."""
    d = _dt.datetime.strptime(date, "%Y%m%d") + _dt.timedelta(days=days)
    return d.strftime("%Y%m%d")


def extract(source, workdir):
    """Return granule paths for one input, extracting a tar if needed.

    The second element of the pair says whether a temporary directory was
    created, so the caller knows what it may remove.
    """
    if os.path.isdir(source):
        return sorted(glob.glob(os.path.join(source, "*.nc"))), None
    if not tarfile.is_tarfile(source):
        raise ValueError(f"{source}: not a directory or tar archive")
    dest = os.path.join(workdir, os.path.basename(source) + ".d")
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(source) as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".nc")]
        try:
            tf.extractall(dest, members=members, filter="data")
        except TypeError:                      # Python < 3.12
            tf.extractall(dest, members=members)
    return sorted(glob.glob(os.path.join(dest, "**", "*.nc"),
                            recursive=True)), dest


class GranuleCache:
    """Extract each archive once, since neighbouring days reuse it."""

    def __init__(self, workdir):
        self.workdir = workdir
        self._paths = {}
        self._temp = {}
        self._key = {}

    def get(self, source, key):
        """Granule paths for an archive, extracting it on first use.

        `key` is the (satellite, date) the archive belongs to, remembered so the
        cache can drop archives no later day can reach.
        """
        if source not in self._paths:
            paths, temp = extract(source, self.workdir)
            self._paths[source] = paths
            self._temp[source] = temp
            self._key[source] = key
        return self._paths[source]

    def release(self, source):
        """Drop one archive."""
        temp = self._temp.pop(source, None)
        self._paths.pop(source, None)
        self._key.pop(source, None)
        if temp:
            shutil.rmtree(temp, ignore_errors=True)

    def retain_from(self, sat, date):
        """Drop every archive before `date`, and every other satellite's.

        Releasing only the archive dated exactly one step back would keep the
        whole run cached whenever the days supplied are not consecutive, since
        no cached archive would ever match.
        """
        for source, (s, d) in list(self._key.items()):
            if s != sat or d < date:
                self.release(source)


def main():
    argv = sys.argv
    if len(argv) < 2:
        print(__doc__)
        return 1
    out_dir = opt(argv, "--out", "../data/f13_1998_tdr_grid")
    force = "--force" in argv
    args = [a for a in argv[1:] if not a.startswith("--") and a != out_dir]
    if not args:
        print(__doc__)
        return 1

    # Keyed by satellite as well as date. Neighbouring archives are offered to
    # each day, so keying on the date alone would let one satellite's orbits be
    # binned into another's grid and the output would carry whichever satellite
    # happened to win the mapping.
    by_day = {}
    for source in args:
        key = day_key(source)
        if key is None:
            print(f"skip {source}: no satellite and date in the name")
            continue
        by_day.setdefault(key, []).append(source)
    if not by_day:
        print("no usable inputs")
        return 1

    os.makedirs(out_dir, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="swi_tdr_")
    cache = GranuleCache(workdir)
    written = []
    short_days = []
    incomplete_days = []
    try:
        for sat, date in sorted(by_day):
            prev_date = shift_day(date, -1)
            # Nothing from here on reaches further back than the previous day,
            # whatever path the last iteration took through this loop.
            cache.retain_from(sat, prev_date)

            out_path = os.path.join(out_dir, f"SWI_TDRGRID_{sat}_D{date}.nc")
            if os.path.exists(out_path) and not force:
                print(f"{sat} {date}: already gridded, use --force to rebuild")
                written.append(out_path)
                continue

            # A granule can be reached through more than one archive when the
            # archives supplied overlap, so offer each one once. A damaged
            # archive costs its own day, not the whole run.
            offered, seen, bad = [], set(), []
            for d in (prev_date, date, shift_day(date, 1)):
                for src in by_day.get((sat, d), ()):
                    try:
                        paths = cache.get(src, (sat, d))
                    except Exception as exc:
                        bad.append((src, f"archive unreadable, "
                                         f"{type(exc).__name__}: {exc}"))
                        continue
                    for path in paths:
                        name = os.path.basename(path)
                        if name not in seen:
                            seen.add(name)
                            offered.append(path)
            if not offered:
                print(f"{sat} {date}: no granules found")
                for src, reason in bad:
                    print(f"    {os.path.basename(src)}: {reason}")
                continue

            lat, lon, means, counts, report = tdr.grid_day(offered, day=date)
            if bad:
                report = report._replace(rejected=report.rejected + bad)
            if not report.used:
                print(f"{sat} {date}: no scans fell in this day")
                continue
            tdr.write_grid(out_path, lat, lon, means, counts, report.used, sat,
                           report=report)
            filled = int((counts[..., 0] > 0).sum())
            note = ""
            if (sat, prev_date) not in by_day:
                short_days.append(f"{sat} {date}")
                note = ", SHORT: no preceding archive"
            if not report.complete:
                note += (f", INCOMPLETE: {report.covered_fraction * 100:.1f}% "
                         f"of the day covered")
                incomplete_days.append(f"{sat} {date}")
            print(f"{sat} {date}: {len(report.used):2d} of {len(offered):2d} "
                  f"granules used, {filled:,} cells with 19V, "
                  f"wrote {os.path.basename(out_path)}{note}")
            for path, reason in report.rejected + report.empty:
                print(f"    refused {os.path.basename(path)}: {reason}")
            for a, b in report.gaps:
                print(f"    no scans {tdr._hhmmss(a)} to {tdr._hhmmss(b)} UTC")
            written.append(out_path)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print(f"\n{len(written)} day(s) gridded into {out_dir}")
    if short_days:
        print(f"{len(short_days)} day(s) built without a preceding archive and "
              f"are missing their first scans: {', '.join(short_days)}")
    if incomplete_days:
        print(f"{len(incomplete_days)} day(s) are marked incomplete, from a "
              f"refusal, a granule without usable data, or a gap in scan "
              f"coverage: {', '.join(incomplete_days)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
