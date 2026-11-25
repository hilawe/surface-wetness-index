"""What the map figures of the manuscript put on their color scales and axes.

Figures 3, 4 and 6 show the wetness index over a reference. The index panel
was once drawn on its full range of 0 to 100, where nearly every plotted cell
is at or under 20, so it showed little. These tests hold the three scripts to
one scale for that panel, to a label that names the quantity and not a range,
to a pointed end on a scale that values run past, and to negative tick labels
set with a hyphen, as every other figure of the paper has them.
"""

import numpy as np
import pytest

from swi import figcache

matplotlib = pytest.importorskip("matplotlib")


@pytest.fixture(autouse=True)
def _plot_settings_left_as_found():
    """Each test starts from the typographic minus and leaves no setting behind.

    The scripts set the hyphen themselves. Started from whatever an earlier
    test left set, a script that had lost that line still drew hyphens here
    and the test passed for the wrong reason.
    """
    with matplotlib.rc_context({"axes.unicode_minus": True}):
        yield
    import matplotlib.pyplot as plt
    plt.close("all")


def _fields():
    rng = np.random.default_rng(3)
    lat = np.linspace(-60.0, 60.0, 24)
    lon = np.linspace(0.0, 345.0, 24)
    wet = rng.random((24, 24)) * 60.0
    ref = rng.random((24, 24)) * 0.8
    mask = np.ones((24, 24), bool)
    return lat, lon, wet, ref, mask


def _draw(which, tmp_path):
    lat, lon, wet, ref, mask = _fields()
    out = str(tmp_path / "map.png")
    if which == "swamps":
        from scripts import validate_swamps as drv
        drv._map(lat, lon, wet, ref, mask, out)
    elif which == "esacci":
        from scripts import validate_esacci as drv
        drv._map(lat, lon, wet, ref, mask, "202307", out)
    else:
        from scripts import validate_smap as drv
        drv._map(lat, lon, wet, ref, mask, "202307", out)
    import matplotlib.pyplot as plt
    return plt.gcf()


@pytest.mark.parametrize("which", ["swamps", "esacci", "smap"])
def test_the_index_panel_is_drawn_on_the_shared_scale(which, tmp_path):
    fig = _draw(which, tmp_path)
    index_panel = fig.axes[0].images[0]
    assert index_panel.get_clim() == (0, figcache.MAP_INDEX_MAX) == (0, 20)
    assert index_panel.colorbar.ax.get_ylabel() == "wetness index"


@pytest.mark.parametrize("which", ["swamps", "esacci", "smap"])
def test_a_scale_that_values_run_past_ends_in_a_point(which, tmp_path):
    """Both panels hold values above their limits in every figure."""
    fig = _draw(which, tmp_path)
    for panel in (fig.axes[0], fig.axes[1]):
        assert panel.images[0].colorbar.extend == "max"


def test_the_inundation_scale_is_not_labeled_with_a_range_it_does_not_span(tmp_path):
    fig = _draw("swamps", tmp_path)
    reference = fig.axes[1].images[0]
    assert reference.get_clim() == (0, 0.5)
    assert reference.colorbar.ax.get_ylabel() == "surface-water fraction"


def _negative_tick_labels(fig):
    fig.canvas.draw()
    labels = [t.get_text() for ax in fig.axes
              for t in ax.get_xticklabels() + ax.get_yticklabels()]
    return [text for text in labels if text[:1] in ("-", "−")]


@pytest.mark.parametrize("which", ["swamps", "esacci", "smap"])
def test_negative_tick_labels_on_a_validation_map_use_a_hyphen(which, tmp_path):
    negative = _negative_tick_labels(_draw(which, tmp_path))
    assert negative and all(text.startswith("-") for text in negative)


def test_negative_tick_labels_on_the_product_map_use_a_hyphen(tmp_path):
    from scripts import make_paper_fig1 as drv
    lat, lon, wet, _ref, _mask = _fields()
    drv._plot(lat, lon, wet, wet + 250.0, wet / 60.0, "202307", "dsc",
              str(tmp_path / "product.png"))
    import matplotlib.pyplot as plt
    negative = _negative_tick_labels(plt.gcf())
    assert negative and all(text.startswith("-") for text in negative)


def test_negative_tick_labels_on_the_temporal_map_use_a_hyphen(tmp_path):
    from scripts import validate_era5_temporal as drv
    lat, lon, wet, _ref, _mask = _fields()
    drv._map(lat, lon, wet / 30.0 - 1.0, str(tmp_path / "temporal.png"))
    import matplotlib.pyplot as plt
    negative = _negative_tick_labels(plt.gcf())
    assert negative and all(text.startswith("-") for text in negative)
