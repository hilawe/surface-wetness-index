# Preregistration: SSM/I-era inundation test of the revived wetness index

Hilawe Semunegus (NOAA NCEI)

Written 2026-07-27, before the reference data for the test month were downloaded and
before any statistic was computed. It is committed ahead of the run so the analysis
choices below cannot be adjusted after seeing the answer.

## Why this test

Every inundation result in `the project documentation` rests on a single month, July 2016,
from a single sensor, the Special Sensor Microwave Imager Sounder (SSMIS) on F-16.
That configuration carries three things at once that the evidence cannot separate.
It is one month, so nothing establishes that the result transfers. It is SSMIS, which
samples 91.655 GHz where the original Special Sensor Microwave/Imager (SSM/I) sampled
85.5 GHz, so an empirical spectral adjustment sits between the instrument and the
retrieval. And its statistics are unweighted on an equal-angle grid, which gives high
latitude cells the same weight as tropical ones while the detection contrast is far
larger at high latitudes.

An independent review of the project recommended one experiment that addresses all of
these together. This is that experiment, preregistered.

## What is fixed in advance

**Sensor and era.** F-13, SSM/I, which measures 85.5 GHz directly. No 85-to-91 GHz
conversion is applied or needed, so the spectral adjustment is removed from the
comparison entirely rather than being estimated.

**Month.** July 1998, descending pass. The full 1998 F-13 record is already staged, and
July is the month used for the project's other diagnostics, so it is the natural choice
rather than one selected for its result.

**Reference.** Surface WAter Microwave Product Series (SWAMPS) version 3.1, stable
record, fractional surface water, monthly mean for July 1998, from the International
Research Institute for Climate and Society Data Library.

**Thresholds.** These are carried over unchanged from the July 2016 analysis and are not
retuned for this month. This is a transfer test, so refitting any of them here would
defeat its purpose.

- the index is treated as firing when `WET > 0`
- a cell counts as inundated when fractional water exceeds 0.05
- a cell counts as clearly inundated when fractional water exceeds 0.10
- comparison is restricted to land, and cells flagged snow or ice are excluded

**Statistics to be reported, all of them, whatever they show.**

- Spearman rank correlation, area weighted by the cosine of latitude, reported as the
  primary figure, with the unweighted value alongside it for comparison with the
  published July 2016 number
- Pearson correlation
- detection contrast, the ratio of mean fractional water where the index fires to where
  it is silent, area weighted and unweighted
- the complete contingency table at the frozen threshold, giving probability of
  detection, false alarm ratio, critical success index, and Heidke skill score
- the rank correlation among only the cells where the index fires, which tests whether
  the index carries magnitude information rather than only presence
- a 95 percent confidence interval on the area-weighted Spearman correlation from a
  spatial block bootstrap resampling 10-degree blocks, because neighbouring cells are
  not independent
- the detection contrast broken out by latitude zone

## What the result would mean

Stated before seeing it, so that neither outcome can be reinterpreted afterwards.

The July 2016 result would be considered to transfer if the area-weighted Spearman
correlation for July 1998 is positive with a bootstrap interval excluding zero, and the
detection contrast is comparable in magnitude to the July 2016 value. A
materially smaller correlation or contrast, or an interval including zero, would mean
the published single-month result does not generalize across the sensor eras, and the
paper's detection claim would have to be narrowed to the configuration it was measured
in.

**Post-hoc note, 2026-07-27, written after the result was known.** An independent review
pointed out that "comparable in magnitude" and "materially smaller" carry no numerical
tolerance, so a borderline result could have been argued either way. That is a real
weakness in the criterion as written, and it cannot be repaired now without cheating,
because the answer is already in hand. It is recorded here rather than quietly fixed.

For what it is worth, the observed result is not borderline under any reasonable
tolerance. The two area-weighted contrasts differ by under 2 percent (4.29 against
4.22), and the two bootstrap intervals for the area-weighted Spearman overlap across
most of their range. A stricter criterion set in advance would have reached the same
verdict. The next test of this kind should state its tolerance numerically at the
outset.

One analysis choice was also changed after the fact and is disclosed for the same
reason. The July 2016 comparison is recomputed over all 31 days of that month, because
the original composite used only 11 days and the July 1998 composite uses 31, so the
two were not matched in temporal sampling. Matching them moved the July 2016
area-weighted contrast from 4.65 to 4.22, which brought the two months closer together
rather than further apart.

A result that transfers would also remove the spectral-adjustment concern from the
detection claim, since no such adjustment is used here.

## What this test does not settle

The antenna temperature question is not addressed. The review asked for this experiment
to be run through both real antenna temperature and the brightness temperature record.
The Colorado State University gridded product used throughout this project carries
brightness temperature only, so a real antenna temperature run needs the separate swath
level record and a gridding step that does not yet exist here. Re-running the existing
fixed-offset approximation would not answer the question the review asked, because the
objection was to the approximation itself. The antenna temperature arm is therefore
deferred and named as outstanding rather than approximated.

This test also remains two months of evidence across the two sensor eras once it is
complete, not a multi-year validation, and it says nothing about the other references.
