"""Grid matched antenna and brightness temperature swaths onto one daily grid.

    python -m scripts.grid_paired_day TAR_OR_DIR [...] --fcdr DIR --out DIR

Each positional input is a daily archive of CSU Temperature Data Record (TDR)
BASE granules, as delivered by NOAA CLASS, or a directory of extracted granules.
`--fcdr` points at a flat directory of CSU Fundamental Climate Data Record
(FCDR) swath granules covering the same dates.

Granules are matched one to one by orbit revision number, and both conventions
are then gridded through a single operator, described in `swi/paired_swath.py`.
The output holds both arms in one file, resting on identical contributing
pixels, so only the radiometric convention differs between them.

Day assembly follows the same rule as `grid_tdr_day`: a granule is filed under
the day it starts, so each day is built from its own archive and the preceding
one, with scan times selecting what belongs. Supply one archive before the range
wanted, or the first day is short.

Options:

    --fcdr DIR   directory of FCDR swath granules (required)
    --out DIR    where to write the paired days
    --force      rebuild days whose output already exists
"""

import datetime as _dt
import glob
import os
import re
import shutil
import sys
import tempfile

from scripts.grid_tdr_day import GranuleCache, day_key, opt, shift_day
from swi import paired_swath as ps


SPAN = re.compile(r"_D(\d{8})_S(\d{2})(\d{2})_E(\d{2})(\d{2})_")


def fcdr_for(fcdr_dir, sat, dates):
    """FCDR granules whose filename names this satellite and one of these dates."""
    out = []
    for d in dates:
        out.extend(sorted(glob.glob(os.path.join(
            fcdr_dir, f"*_{sat}_D{d}_*.nc"))))
    return out


def touches_day(path, date):
    """Whether a granule's filename span reaches into the target UTC day.

    A granule is filed under the day it starts, so one filed on the day after
    the target cannot reach back into it, and one filed on the day before
    reaches in only when it crosses midnight. Used to decide whether a granule
    without a counterpart in the other product actually costs this day any
    coverage, rather than reporting every neighbouring granule as a loss.
    """
    m = SPAN.search(os.path.basename(path))
    if not m:
        return True                     # unparseable, so assume it matters
    filed, sh, sm, eh, em = m.group(1), *map(int, m.groups()[1:])
    start = _dt.datetime.strptime(filed, "%Y%m%d") + _dt.timedelta(hours=sh,
                                                                  minutes=sm)
    end = start.replace(hour=eh, minute=em)
    if end <= start:                    # the span wraps past midnight
        end += _dt.timedelta(days=1)
    day0 = _dt.datetime.strptime(date, "%Y%m%d")
    return start < day0 + _dt.timedelta(days=1) and end > day0


def main():
    argv = sys.argv
    fcdr_dir = opt(argv, "--fcdr")
    out_dir = opt(argv, "--out", "../data/f13_1998_paired_grid")
    force = "--force" in argv
    args = [a for a in argv[1:]
            if not a.startswith("--") and a not in (fcdr_dir, out_dir)]
    if not args or not fcdr_dir:
        print(__doc__)
        return 1
    if not os.path.isdir(fcdr_dir):
        print(f"--fcdr {fcdr_dir}: not a directory")
        return 1

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
    workdir = tempfile.mkdtemp(prefix="swi_paired_")
    cache = GranuleCache(workdir)
    written, incomplete, lopsided = [], [], []
    try:
        for sat, date in sorted(by_day):
            prev_date = shift_day(date, -1)
            cache.retain_from(sat, prev_date)
            window = (prev_date, date, shift_day(date, 1))

            out_path = os.path.join(out_dir, f"SWI_PAIRED_{sat}_D{date}.nc")
            if os.path.exists(out_path) and not force:
                print(f"{sat} {date}: already gridded, use --force to rebuild")
                written.append(out_path)
                continue

            tdr, seen, bad = [], set(), []
            for d in window:
                for src in by_day.get((sat, d), ()):
                    try:
                        paths = cache.get(src, (sat, d))
                    except Exception as exc:
                        bad.append((src, f"archive unreadable, "
                                         f"{type(exc).__name__}: {exc}"))
                        continue
                    for p in paths:
                        if os.path.basename(p) not in seen:
                            seen.add(os.path.basename(p))
                            tdr.append(p)
            fcdr = fcdr_for(fcdr_dir, sat, window)

            pairs, unmatched = ps.pair_paths(tdr, fcdr)
            if not pairs:
                print(f"{sat} {date}: no matched granule pairs")
                for src, reason in bad:
                    print(f"    {os.path.basename(src)}: {reason}")
                continue

            (lat, lon, means_ta, means_tb,
             counts, report) = ps.grid_day_paired(pairs, day=date)
            extra = list(bad)
            # A granule present in one product and not the other would give the
            # two arms different coverage, which is the whole thing this guards.
            for p in unmatched:
                if touches_day(p, date):
                    extra.append((p, "no counterpart in the other product"))
            if extra:
                report = report._replace(rejected=report.rejected + extra)
            if not report.used:
                print(f"{sat} {date}: no scans fell in this day")
                continue

            ps.write_paired_grid(out_path, lat, lon, means_ta, means_tb, counts,
                                 report.used, sat, report=report)
            filled = int((counts[..., 0] > 0).sum())
            note = ""
            if not report.complete:
                note = (f", INCOMPLETE: "
                        f"{report.covered_fraction * 100:.1f}% of the day")
                incomplete.append(f"{sat} {date}")
            if any(touches_day(p, date) for p in unmatched):
                lopsided.append(f"{sat} {date}")
            print(f"{sat} {date}: {len(report.used):2d} of {len(pairs):2d} "
                  f"pairs used, {filled:,} cells with 19V, "
                  f"wrote {os.path.basename(out_path)}{note}")
            for p, reason in report.rejected + report.empty:
                print(f"    {os.path.basename(p)}: {reason}")
            for a, b in report.gaps:
                print(f"    no scans {ps._hhmmss(a)} to {ps._hhmmss(b)} UTC")
            written.append(out_path)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print(f"\n{len(written)} paired day(s) written into {out_dir}")
    if incomplete:
        print(f"{len(incomplete)} day(s) marked incomplete: "
              f"{', '.join(incomplete)}")
    if lopsided:
        print(f"{len(lopsided)} day(s) had a granule in one product only: "
              f"{', '.join(lopsided)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
