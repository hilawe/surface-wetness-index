# Preregistration: an optically independent check of the surface-water claim

Hilawe Semunegus (NOAA NCEI)

Written 2026-07-29, before any reference tile was downloaded and before any statistic
was computed. Committed ahead of the run so the regions, the metric, and the criterion
cannot be adjusted after seeing the answer.

## Why this test

The paper's surviving central claim is that the revived index has a strong spatial
correspondence with surface water. All of the evidence for it comes from one reference,
the Surface WAter Microwave Product Series (SWAMPS), which is built from combined active
and passive microwave observations. The index is a passive-microwave retrieval. Two
independent reviews made the same point: a passive-microwave reference and a
passive-microwave retrieval can agree partly because they share a measurement family and
its error structure, so the agreement is not independent confirmation.

Section 2.2 of the paper already concedes this by classifying SWAMPS as
target-appropriate rather than physically independent. What has been missing is a
reference from a different observation family entirely. This test supplies one.

## The reference, fixed in advance

The Joint Research Centre Global Surface Water dataset (Pekel and colleagues), version
1.4, 2021 release, water occurrence layer. It is derived from the Landsat optical
archive at 30 m resolution and reports, per pixel, the percentage of valid observations
between 1984 and 2021 in which surface water was detected. Optical Landsat shares no
instrument, no frequency, and no retrieval physics with a passive-microwave index, so
agreement between the two cannot come from a shared measurement family.

The layer is a long-record occurrence statistic rather than a single month. That suits
the claim under test, which is about where the index finds water rather than when, and
it is the reason this test addresses the spatial claim and not the event claim
abandoned in section 5.6.

## Regions, fixed in advance

Five 10-degree tiles, chosen to span the regimes across which the paper's latitude
breakdown reports very different behaviour, so the test cannot be rescued by picking
favourable ground.

| Tile | Region | Why |
|------|--------|-----|
| 90E_30N | Ganges and Brahmaputra, Southeast Asia | humid tropics and monsoon floodplain |
| 60E_70N | Western Siberia | boreal, where the reported detection contrast is largest |
| 100W_50N | North American prairie and pothole region | temperate, many small water bodies |
| 0E_30N | Sahara | arid, where the index should find almost nothing |
| 70W_0N | Amazon | humid tropics and dense vegetation |

## Method, fixed in advance

The 30 m occurrence layer is aggregated to the product's grid by averaging occurrence
over each grid cell, giving a per-cell long-term water fraction between 0 and 1. That
fraction is compared with the revived index for July 2016 from F-16, the month already
used in section 5.1, restricted to land and to cells where both are valid.

Reported for each tile and pooled across tiles, all of it whatever it shows:

- Spearman and Pearson correlation between the index and the optical water fraction,
  unweighted and weighted by the cosine of latitude
- the detection contrast, the ratio of mean optical water fraction where the index fires
  to where it is silent
- the full contingency table at the frozen thresholds used elsewhere in the paper, index
  firing at any positive value and the reference counted as inundated above 0.05
- the same statistics computed against SWAMPS on exactly the same cells, so that the two
  references are compared on identical ground rather than across different domains

## What the result would mean

Stated before it is known.

The independent reference would be considered to support the paper's spatial claim if
the index correlates positively with the optical water fraction, with an area-weighted
Spearman correlation at least half the value it achieves against SWAMPS on the same
cells, and a detection contrast above two. Substantially weaker agreement against the
optical reference than against SWAMPS would mean that part of the published agreement
comes from the shared microwave measurement family, and the paper's validation claim
would have to be narrowed accordingly.

A negative or near-zero correlation against the optical reference would be a serious
result for the project, and it would be reported as prominently as a positive one.

## What this test does not settle

The reference is a multi-decade occurrence climatology, so this tests where the index
finds water and not whether it tracks change. Landsat optical detection has its own
biases, in particular under cloud, under dense canopy, and over very small or turbid
water bodies, and those biases differ by region rather than being uniform. Five tiles
are a sample of the globe and not the globe. The comparison also crosses spatial scales,
from 30 m to a third of a degree, and aggregation by averaging is one defensible choice
among several.
