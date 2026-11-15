"""Figure labels for the SWAMPS comparison driver.

`scripts/validate_swamps.py` draws the inundation figure and ships in only
one published subset of this repository, so this file travels with it and is
on that allowlist alone.

It was split out of tests/test_validate.py, which covers the shared
`swi.validate` metrics and therefore ships everywhere. Left there, this test
failed an exported tree that does not contain the driver at all. Making it
skip on a missing import would have fixed that symptom and traded away the
guarantee, because the same skip fires when the driver is present and simply
broken. Moving it keeps it mandatory wherever the driver ships.
"""

from scripts.validate_swamps import _wet_label


def test_the_swamps_panel_label_matches_the_other_validators():
    """Figure 3's panel (a) must name the same quantity the same way.

    The ESA CCI, SMAP and ERA5 figures label it "SWI WET (F-16, YYYYMM)" while
    this one said plain "SWI WET", so a reader meeting Figures 3 and 4 had to
    work out they showed the same field. The sensor and month are derived from
    the product the figure was built from, not hardcoded, because this script
    also draws the F-13 1998 comparison.
    """
    assert _wet_label("SWI_F16_201607_monthly_v0cal.nc") == "SWI WET (F-16, 201607)"
    assert _wet_label("SWI_F13_199807_monthly_v0cal.nc") == "SWI WET (F-13, 199807)"
    # An unrecognizable name must not produce a half-built label naming the
    # wrong sensor; it falls back to the bare quantity.
    for junk in ("", None, "odd_name.nc", "SWI_F16_monthly.nc"):
        assert _wet_label(junk) == "SWI WET"
