"""The valid-range sensitivity the SMAP validation driver stores.

Section 5.3 of the manuscript says that removing daily retrievals outside
the product's stated range changes the comparison by 51 cells and by nothing
at the precision reported. The driver writes that comparison beside the
baseline, so the sentence has a result file behind it.
"""

import numpy as np
import pytest

from scripts import validate_smap as drv
from swi import validate as val


def _fields():
    rng = np.random.default_rng(7)
    plat = np.linspace(-58.0, 58.0, 30)
    wet = np.where(rng.random((30, 40)) < 0.4, rng.random((30, 40)) * 20.0, 0.0)
    sm = 0.05 + 0.3 * rng.random((30, 40)) + 0.004 * wet
    snowf = np.zeros((30, 40))
    return plat, wet, sm, snowf


def test_the_sensitivity_scores_the_same_comparison_on_the_screened_field():
    plat, wet, sm, snowf = _fields()
    strict = sm.copy()
    strict[3, 5] = strict[10, 2] = np.nan       # cells the range screen emptied
    snowf[0, 0] = 1.0                            # frozen, in neither population
    stats = {"valid_range": {"valid_min": 0.02, "valid_max": 0.5,
                             "units": "cm**3/cm**3", "fine_kept": 900,
                             "fine_rejected": 7}}
    out = drv.range_sensitivity(wet, snowf, plat, strict, stats, baseline_n=1199)

    m = np.isfinite(strict) & ~(snowf > 0.5)
    assert out["n_cells"] == int(m.sum()) == 1197
    assert out["n_cells_baseline"] == 1199
    assert out["cells_removed"] == 2
    assert out["fine_retrievals_rejected"] == 7
    assert (out["valid_min"], out["valid_max"], out["units"]) == (
        0.02, 0.5, "cm**3/cm**3")
    latm = np.broadcast_to(plat[:, None], wet.shape)[m]
    s = val.skill_scores(wet[m], strict[m])
    assert out["spearman"] == pytest.approx(s["spearman_r"])
    assert out["pearson"] == pytest.approx(s["pearson_r"])
    assert out["detection_contrast_ratio"] == pytest.approx(
        val.detection_contrast(wet[m], strict[m], thr=0.0)["ratio"])
    assert out["area_weighted"] == val.area_weighted_scores(
        wet[m], strict[m], latm, thr=0.0)


def test_the_sensitivity_refuses_a_composite_made_without_the_range_screen():
    plat, wet, sm, snowf = _fields()
    with pytest.raises(ValueError, match="valid_range"):
        drv.range_sensitivity(wet, snowf, plat, sm, {"screen": "recommended"},
                              baseline_n=1200)
