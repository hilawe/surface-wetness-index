# Preregistration: does the wetness index follow flood events?

Hilawe Semunegus (NOAA NCEI)

Written 2026-07-29, before the reference data for the test months were downloaded and
before any statistic was computed. Committed ahead of the run so the event, the
regions, the metric, and the pass criterion cannot be adjusted after seeing the answer.

## Why this test

Every inundation result in `the project documentation` compares whole monthly spatial
fields. A high correlation of that kind can be produced by permanent water bodies,
lakes, and wetlands whose positions do not change, so it shows the index knows where
water is. It does not show that the index registers water arriving and leaving, which
is the property an operational monitor would need. An independent review made this
point, and section 6.2 of the paper now concedes it. This test addresses it directly.

## The events, fixed in advance

Two large, independently documented 1998 floods, chosen because the F-13 record for
all of 1998 is already staged and because both are large enough to resolve at a third
of a degree.

- **Ganges and Brahmaputra basin, Bangladesh and eastern India.** The 1998 flood
  inundated roughly two thirds of Bangladesh. Onset in July, peak in September,
  recession through October. Region box 20 to 27 degrees north, 86 to 93 degrees east.
- **Yangtze basin, central China.** The 1998 flood was among the most severe of the
  century, rising through June and July to a peak in August. Region box 26 to 33
  degrees north, 108 to 122 degrees east.

Both boxes are stated here as fixed. They will not be moved, grown, or shrunk after
the result is seen.

## Method, fixed in advance

Twelve monthly composites of the revived index for 1998 from F-13, descending pass,
all available days per month, no spectral adjustment because F-13 measures 85.5 GHz
directly. Twelve matching monthly means of SWAMPS fractional surface water.

The comparison is on anomalies, not on absolute fields, because absolute fields are
what carries the permanent-water confound. For each cell, the 1998 annual mean is
subtracted from both series, leaving each month as a departure from that cell's own
year. Permanent water is close to constant across the year and therefore leaves the
anomaly comparison almost entirely.

Reported for each region, all of it whatever it shows:

- the monthly regional-mean anomaly series for both the index and the reference
- the month in which each series peaks
- the correlation between the two regional-mean anomaly series across the twelve months
- the median per-cell temporal correlation of the two anomaly series within the region
- the fraction of cells with a positive per-cell anomaly correlation
- the same statistics for a control region of similar size and latitude with no
  1998 flood on record, to show the metric is not simply positive everywhere

## What the result would mean

Stated before it is known.

The index would be considered to follow the events if, in both flood regions, the
regional-mean anomaly series of the index and the reference peak within one month of
each other and their correlation across the twelve months is positive. A peak more
than one month apart, or a non-positive correlation, in either region would mean the
index does not track this flood at monthly resolution, and the paper's language would
have to stay at spatial correspondence with no event claim at all.

A positive result would be the first evidence in this project that the index responds
to water arriving and leaving rather than only to where water sits. It would still be
two events in one year at monthly resolution, which is not a demonstration of
operational flood-monitoring skill, and it would not license that claim.

## Outcome, recorded after the fact

Added 2026-07-29, after the result was known, and kept with the original text above
unaltered.

The criterion written above was met in both flood regions, and it should not have been
enough. An independent review pointed out that subtracting each cell's annual mean
removes the permanent component but leaves the seasonal cycle untouched, because a
correlation does not change when a constant is subtracted. A July-to-September box-car
carrying no information about 1998 correlates with the index series at +0.87 over the
Ganges and Brahmaputra, slightly better than the +0.86 the index scores against the
reference. Both series are tracking the wet season, and the basin floods every year, so
the test cannot distinguish the 1998 event from ordinary seasonality.

The criterion had a second weakness. It asked only that the two series peak within one
month of each other, and did not require the peak to fall near the documented flood. The
Yangtze region therefore passed on an April peak although August was named above as the
expected month. A criterion that had used the expected months would have failed that
region outright.

The preregistration is left as written, and section 5.6 reports the negative result. The
lesson carried forward is that an anomaly must be taken against a climatology built from
other years, not against the same year's own mean, and that a seasonality control belongs
in the design rather than in the review.

## What this test does not settle

Monthly composites cannot resolve the timing of a flood crest to better than a month,
so nothing here speaks to timing error in the sense an operational product would
report. The test also uses the same reference family as the rest of the inundation
work, so it inherits the shared passive-microwave error structure noted in section
2.2 of the paper. It says nothing about drought or drying outside these two cases.
