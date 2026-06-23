"""USCRN loader: monthly aggregation, fill handling, and min-days threshold."""

from swi import io_uscrn as io


def _row(date, lon, lat, sm):
    f = ["-9999.0"] * 28
    f[io.WBAN_COL] = "54321"
    f[io.DATE_COL] = date
    f[io.LON_COL] = f"{lon}"
    f[io.LAT_COL] = f"{lat}"
    f[io.SM5_COL] = f"{sm}"
    return " ".join(f)


def test_load_station_monthly(tmp_path):
    lines = []
    # 12 valid July days at 0.25 -> kept, mean 0.25
    for d in range(1, 13):
        lines.append(_row(f"202307{d:02d}", -100.0, 40.0, 0.25))
    # one July fill day -> skipped, does not change the mean
    lines.append(_row("20230713", -100.0, 40.0, io.FILL))
    # only 5 June days -> dropped by min_days
    for d in range(1, 6):
        lines.append(_row(f"202306{d:02d}", -100.0, 40.0, 0.30))
    (tmp_path / "CRND0103-2023-XX_Test.txt").write_text("\n".join(lines) + "\n")

    out = io.load_station_monthly(str(tmp_path), min_days=10)
    assert set(out) == {"54321"}
    st = out["54321"]
    assert st["lon"] == -100.0 and st["lat"] == 40.0
    assert "202307" in st["months"] and "202306" not in st["months"]
    assert abs(st["months"]["202307"] - 0.25) < 1e-9
