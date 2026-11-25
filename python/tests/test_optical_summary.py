"""The two summaries the optical independence driver stores.

Table 4 of the manuscript prints the median over the four tiles with a
signal. The driver stored only the median over all five, the arid control
included, so the table's summary row had no named value in the result file
and was once filled in from the wrong one.
"""

import json

import pytest

from scripts import optical_independence_test as opt


def _tiles(optical, swamps):
    names = ["90E_30N", "60E_70N", "100W_50N", "70W_0N", "0E_30N"]
    return {n: {"vs_optical": {"spearman_area_weighted": o},
                "vs_swamps": {"spearman_area_weighted": s}}
            for n, o, s in zip(names, optical, swamps)}


TILES = _tiles([0.47, 0.71, 0.58, 0.56, 0.00], [0.75, 0.93, 0.70, 0.81, -0.01])


def test_the_five_tile_median_includes_the_control():
    out = opt.tile_medians(TILES)
    assert out["vs_optical"] == pytest.approx(0.56)
    assert out["vs_swamps"] == pytest.approx(0.75)
    assert out["ratio"] == pytest.approx(0.56 / 0.75)


def test_the_signal_median_is_over_the_four_tiles_table_4_names():
    out = opt.signal_tile_median(TILES)
    assert out["n_tiles"] == 4
    assert out["tiles"] == ["100W_50N", "60E_70N", "70W_0N", "90E_30N"]
    assert out["vs_optical"] == pytest.approx(0.57)
    assert out["vs_swamps"] == pytest.approx(0.78)
    assert opt.CONTROL_TILE not in out["tiles"]


def test_the_signal_median_is_refused_when_a_registered_tile_is_missing():
    short = {k: v for k, v in TILES.items() if k != "70W_0N"}
    with pytest.raises(ValueError, match="70W_0N"):
        opt.signal_tile_median(short)


def _stored(tmp_path, tiles=TILES, median=None):
    path = tmp_path / "optical.json"
    doc = {"product": "p.nc", "tiles": tiles,
           "within_tile_median": median or opt.tile_medians(tiles)}
    path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    return path, doc


def test_a_stored_result_gains_the_signal_median_and_nothing_else_moves(tmp_path):
    path, before = _stored(tmp_path)
    assert opt.resummarize(str(path), str(path)) == 0
    after = json.loads(path.read_text())
    assert after.pop("signal_tile_median") == opt.signal_tile_median(TILES)
    assert after == before


def test_a_stored_result_whose_summary_does_not_follow_from_its_tiles_is_refused(tmp_path):
    """The recomputation has to reproduce what the full run stored."""
    path, _ = _stored(tmp_path, median={"vs_optical": 0.5, "vs_swamps": 0.75,
                                        "ratio": 0.5 / 0.75})
    text = path.read_text()
    with pytest.raises(ValueError, match="within_tile_median"):
        opt.resummarize(str(path), str(path))
    assert path.read_text() == text
