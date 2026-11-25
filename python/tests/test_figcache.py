"""The save resolution of the figure scripts, and the threshold figure's redraw.

The figures kept with the repository are screen copies at 150 dots per inch.
A journal revision needs the same figures at print resolution, so the
resolution is one setting every figure script reads.
"""

import json

import pytest

from swi import figcache


@pytest.fixture(autouse=True)
def _plot_settings_left_as_found():
    """Drawing a figure here must not change how a later test draws one.

    The threshold figure turns off the typographic minus sign for the whole
    process. Left set, it changed the tick labels of the map figures that the
    reproducibility test redraws afterward, and five of them stopped matching
    their committed copies.
    """
    try:
        import matplotlib
    except ImportError:
        yield
        return
    with matplotlib.rc_context():
        yield


def test_the_default_is_the_screen_resolution_of_the_committed_figures(monkeypatch):
    monkeypatch.delenv(figcache.DPI_ENV, raising=False)
    assert figcache.save_dpi() == 150


def test_the_setting_names_another_resolution(monkeypatch):
    monkeypatch.setenv(figcache.DPI_ENV, "600")
    assert figcache.save_dpi() == 600


def test_an_empty_setting_is_the_default(monkeypatch):
    monkeypatch.setenv(figcache.DPI_ENV, "  ")
    assert figcache.save_dpi() == 150


@pytest.mark.parametrize("value", ["print", "600.5", "30", "5000", "-600"])
def test_a_setting_that_is_no_usable_resolution_is_refused(monkeypatch, value):
    """A typo must not fall back to the screen copies in silence."""
    monkeypatch.setenv(figcache.DPI_ENV, value)
    with pytest.raises(ValueError):
        figcache.save_dpi()


def _result(tmp_path):
    def sweep(shift):
        curve = [{"threshold": t, "POD": 0.9 - 0.1 * t - shift,
                  "precision": 0.3 + 0.1 * t} for t in range(4)]
        return {"curve": curve, "best_csi": curve[2], "n": 10}
    path = tmp_path / "threshold.json"
    path.write_text(json.dumps({"F13_July1998": sweep(0.05),
                                "F16_July2016": sweep(0.0),
                                "thresholds": [0, 1, 2, 3]}))
    return str(path)


def test_the_threshold_figure_redraws_from_its_result_file(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from scripts import threshold_transfer_test as thr
    monkeypatch.delenv(figcache.DPI_ENV, raising=False)
    first = tmp_path / "a.png"
    thr.redraw(_result(tmp_path), str(first), ["F16_July2016", "F13_July1998"])
    again = tmp_path / "b.png"
    thr.redraw(_result(tmp_path), str(again), ["F16_July2016", "F13_July1998"])
    assert first.read_bytes() == again.read_bytes()
    # the order of the two sweeps decides the colors, so it changes the picture
    other = tmp_path / "c.png"
    thr.redraw(_result(tmp_path), str(other), ["F13_July1998", "F16_July2016"])
    assert first.read_bytes() != other.read_bytes()


def test_the_threshold_redraw_refuses_labels_that_are_not_its_sweeps(tmp_path):
    pytest.importorskip("matplotlib")
    from scripts import threshold_transfer_test as thr
    with pytest.raises(SystemExit):
        thr.redraw(_result(tmp_path), str(tmp_path / "x.png"),
                   ["F16_July2016", "F18_July2023"])


def test_a_figure_saved_at_the_setting_is_that_many_times_larger(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from scripts import threshold_transfer_test as thr

    def width(path):
        return int.from_bytes(open(path, "rb").read(24)[16:20], "big")
    monkeypatch.delenv(figcache.DPI_ENV, raising=False)
    thr.redraw(_result(tmp_path), str(tmp_path / "screen.png"))
    monkeypatch.setenv(figcache.DPI_ENV, "600")
    thr.redraw(_result(tmp_path), str(tmp_path / "print.png"))
    assert width(tmp_path / "screen.png") == 1125
    assert width(tmp_path / "print.png") == 4500
