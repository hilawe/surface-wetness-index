# Scope: an L-band SMAP comparison for the reconstruction paper

Status: SCOPED AND RUN 2026-08-26. The comparison is built and executed, and the
outcome is recorded below. The pre-registered expectation was NOT confirmed. What
remains is manuscript integration and an independent review pass.

## Why

The manuscript compares the wetness index against SWAMPS inundation, ESA CCI COMBINED
soil moisture, ERA5-Land, and USCRN stations. It does not mention L-band, SMAP, or SMOS
anywhere. Two separate problems follow from that, and one addition fixes both.

The first is an independence problem the paper already admits. Its own data section says
the ESA CCI COMBINED product "merges passive-microwave retrievals from many of the same
DMSP imager channels the wetness index uses, and shared sensors and shared seasonal
microwave artifacts can inflate apparent agreement." The paper proposes an ACTIVE-only
C-band run as the remedy.

The second is a positioning problem. The most damaging criticism available against this
work is that it reconstructs a legacy high-frequency method while ignoring the dedicated
L-band missions that are the modern standard for soil moisture. A reviewer can raise it,
and so can a skeptical manager.

SMAP answers both. At 1.4 GHz it shares no channel, no sensor, and no artifact family
with a 19 to 85 GHz index, so it is a genuinely independent reference, which the C-band
extension also would be. Unlike the C-band extension it also demonstrates that the paper
tested against the current standard and reported where the two differ.

## Pre-registered expectation, recorded before anything is run

**Agreement with SMAP is expected to be LOWER than agreement with ESA CCI, and that is
the correct result, not a failure.** The reported ESA CCI agreement is partly inflated by
shared DMSP channels. Removing that shared-sensor path should reduce apparent skill. The
committed prediction is that the detection contrast against SMAP stays above 1.0 with a
bootstrap interval excluding 1.0, while falling below the 1.43 measured against ESA CCI.

This is written down now so the outcome cannot be rationalized afterwards. If the contrast
against SMAP comes out higher than against ESA CCI, that is a surprise and needs an
explanation, not a victory lap. If it fails to exclude 1.0, that is a genuine negative
result about the index and belongs in the paper as one.

## OUTCOME, run 2026-08-26. The prediction above was NOT confirmed.

The expectation was that agreement with SMAP would fall below the 1.43 detection
contrast measured against ESA CCI. That did not happen. The index agrees BETTER with the
independent L-band reference, on every metric, and the difference survives every check
run against it.

| domain | reference | n cells | Spearman | contrast | 95 percent interval |
|---|---|---|---|---|---|
| own | SMAP L-band | 144,560 | +0.618 | 2.07 | 1.87 to 2.30 |
| own | ESA CCI COMBINED | 193,621 | +0.353 | 1.43 | 1.30 to 1.56 |
| matched | SMAP L-band | 138,120 | +0.630 | 2.08 | 1.87 to 2.31 |
| matched | ESA CCI COMBINED | 138,120 | +0.509 | 1.61 | 1.48 to 1.77 |

The matched rows score both references on the SAME 138,120 cells, which removes the
possibility that SMAP simply kept an easier domain. Restricting ESA CCI to those cells
does raise its contrast from 1.43 to 1.61, so domain selection explains part of the raw
gap. Domain selection does not explain the rest, and on matched cells the intervals do
not overlap.

Two robustness checks were run because a surprise in a flattering direction deserves more
suspicion than a disappointing one. Dropping the quality screen entirely gives a contrast
of 2.02 over a larger 235,555 cells, and replacing the area-binning with nearest-neighbour
sampling gives 2.09. Neither processing choice is manufacturing the result.

WHY THE PREDICTION WAS WRONG, which matters more than the number. The reasoning was that
shared DMSP channels inflate the ESA CCI agreement, so removing the shared path would
lower apparent skill. That conflated two different statistics. Shared instrument noise can
inflate a CORRELATION between fields that carry it, but the detection contrast asks
whether soil moisture is genuinely higher where the index fires, and shared noise does not
manufacture that relationship, it blurs it. For this statistic the QUALITY of the
reference matters more than its INDEPENDENCE, and SMAP is a purpose-built L-band soil
moisture retrieval with less vegetation attenuation. The shared-sensor caveat about ESA
CCI still stands as a reason not to call it an independent reference. The evidence does
not support it as a reason to think the CCI agreement was inflated, and the manuscript
should stop implying that it was.

THE CAVEATS THAT SURVIVE, and they are not small. The SMAP quality screen keeps 24.9
percent of the cells where the index fires against 31.0 percent of the cells where it is
quiet, a retention ratio of 0.80, so firing cells are preferentially screened out and the
comparison runs on ground where SMAP is confident. Pearson correlation is only +0.283
against a Spearman of +0.618, so the relationship is monotonic rather than linear, which
is consistent with the index being a rank indicator rather than a soil moisture estimate
and should be stated that way. The contrast is strong in the tropics at 2.57 and weak at
high latitudes at 1.20. This remains one month, one satellite, one product version.

Artifacts are `results/validate_smap_F16_202307.json`,
`results/smap_cci_matched_F16_202307.json`, and `docs/figures/fig_smap.png`, all
regenerable from committed scripts.

## Data

- Product: SMAP Enhanced L3 Radiometer Global Daily Soil Moisture (SPL3SMP_E), 9 km
  EASE-Grid 2.0, latest available version. The 9 km enhanced product is chosen over the
  36 km standard product because 36 km is coarser than the 0.25 degree target grid and
  would smear the comparison. The files carry their own latitude and longitude arrays,
  so no EASE-Grid projection maths is required, which removes the main technical risk.
- Overpass: the AM (descending, about 6 am local) retrieval, matched to the wetness
  index descending pass. Both are early morning and near thermal equilibrium, so this is
  a fair pairing rather than a convenient one. The PM pairing is available as a
  sensitivity check and is not the headline.
- Month: July 2023, the same month already used for the ESA CCI and ERA5-Land
  comparisons, so the new number sits directly alongside the existing ones in the same
  table with no confound from a different period.
- Access: VERIFIED WORKING 2026-08-26. The `urs.earthdata` entry in `~/.netrc` (mode 600)
  authenticates successfully against the protected NSIDC Earthdata Cloud endpoint. A
  byte-range probe of a real July 2023 granule returned HTTP 206 with the HDF5 magic
  number intact, so the whole redirect and authentication chain completes and the bytes
  are genuine product. No credential refresh is needed and this is no longer a blocker.
- Confirmed particulars, from the public metadata search rather than assumption. The
  collection is `SPL3SMP_E` version **006**, served from
  `data.nsidc.earthdatacloud.nasa.gov`, with granules named like
  `SMAP_L3_SM_P_E_20230715_R19240_001.h5`. July 2023 coverage is complete, 31 days with
  no gap.
- SUBSETTING WORKS AND THE VOLUME PROBLEM IS SOLVED, verified end to end 2026-08-26. A
  full granule is about 629 MB, so a bulk month would be roughly 19.5 GB. The collection
  (concept id `C2938664763-NSIDC_CPRD`) exposes an OPeNDAP endpoint at
  `opendap.earthdata.nasa.gov`, and a variable-constrained request for just the AM soil
  moisture field and its retrieval quality flag returns **3.75 MB**, which is about
  116 MB for the whole month. That is a 99 percent reduction and removes the transfer
  cost as a consideration.
- THE WORKING REQUEST FORM, which took three attempts to find and should not be
  rediscovered later. The response suffix must be **`.dap.nc4`**, which returns compressed
  netCDF4. Both `.nc4` and `.nc` return HTTP 400 from this server. Metadata comes from
  `.dmr`. The constraint expression is `dap4.ce=` with fully qualified variable paths
  separated by an encoded semicolon, for example
  `/Soil_Moisture_Retrieval_Data_AM/soil_moisture` and
  `/Soil_Moisture_Retrieval_Data_AM/retrieval_qual_flag`. Authentication is the same
  netrc plus cookie jar chain used for the direct download.
- THE RETURNED SUBSET WAS OPENED AND CHECKED, not merely counted. It carries the AM soil
  moisture as float32 and the quality flag as uint16, both on the expected 1624 by 3856
  EASE-Grid 2.0 9 km global grid, 6,262,144 cells. Soil moisture spans 0.020 to 0.859
  cubic metres per cubic metre with a mean of 0.216, which is a physically sane volumetric
  range. On that single day 844,552 cells carry a valid retrieval and 461,210 of those
  meet the recommended quality flag.
- ONE IMPLEMENTATION DETAIL THE PROBE EXPOSED. The constrained netCDF response returns
  index dimensions rather than coordinates, so the `latitude` and `longitude` arrays must
  be requested explicitly. They describe a fixed EASE grid, so the fetcher should retrieve
  them once and reuse them rather than pulling them with every daily file.

## Method, mirroring the existing harness exactly

`python/scripts/validate_esacci.py` is the template and the new script follows it step for
step so the numbers are comparable rather than merely adjacent.

- Regrid the SMAP field to the product grid with `validate.regrid_nearest`.
- Mask to common-valid land, unfrozen (`snowf <= 0.5`), and real wetness (`wet >= 0`),
  then additionally apply SMAP's own recommended retrieval quality flag.
- Report the same statistics: whole-field Spearman and Pearson, pattern correlation, the
  detection contrast (mean soil moisture where the index fires against where it does not,
  and the ratio), Spearman among firing cells, the categorical scores against the upper
  soil-moisture tercile, and the contrast by latitude zone.
- Confidence interval by the same spatial block bootstrap, 10 degree blocks, 2000 draws,
  with the same seed convention and a null value of 1.0.
- Emit the same JSON result record and a cached figure.

## The one methodological risk worth naming up front

SMAP screens aggressively. Its quality flags remove dense vegetation, frozen ground,
urban areas, and open water, and open water and very wet surfaces are part of where the
wetness index is most active. The common-valid domain could therefore exclude the very
cells that carry the index's signal, which would make the comparison look worse than the
physics warrants and for an uninteresting reason.

This must be measured rather than assumed. The script reports the retained cell count and
the fraction of firing cells that survive the SMAP mask, and the paper states both. If
the retained fraction is low, the honest conclusion is that SMAP and this index have
limited common ground, which is itself a publishable and useful statement about what the
index measures.

## Deliverables

1. `python/swi/io_smap.py`, a loader mirroring `io_esacci.py`.
2. `python/scripts/fetch_smap.py`, a fetcher with validation, mirroring
   `fetch_swamps.py`, so this input never becomes an unscripted casualty.
3. `python/scripts/validate_smap.py`, mirroring `validate_esacci.py`.
4. `results/validate_smap_202307.json`, the committed numbers.
5. `docs/figures/fig_smap.png` plus its plotting-array cache and an entry in
   `docs/figures/figure_provenance.json`, so it passes the reproducibility guard.
6. Unit tests for the loader and an entry in `test_figure_reproducibility.py`.
7. Manuscript edits, meaning a data-section entry, a results subsection, a new row in the
   validation table, and a sentence in the discussion positioning the index against
   L-band retrieval.
8. One adversarial review pass from a different model family over the new code and the
   new claims, per the project's standing practice.

## Effort and sequencing

Roughly two focused sessions, and both access questions are now settled at no cost. One
session covers the fetcher, the loader, and the comparison, ending with committed numbers.
A second covers the figure, the manuscript integration, and the review round.

Only one place could still run long. That is the masking-domain question above, which may
need a sensitivity run before the result can be stated cleanly. The probe already gives an
early reading on it, since the recommended quality flag keeps 461,210 of 844,552 valid
retrievals on a single day, so roughly 45 percent are discarded before the comparison even
begins. Whether the survivors overlap the cells where the index actually fires is the
thing to measure first, ahead of any statistic.

## Decisions needed before starting

1. **Replace or add?** Recommendation is that SMAP REPLACES the ACTIVE-only C-band
   extension the paper currently promises, rather than the paper carrying both. SMAP does
   the independence job the C-band run was for and answers the modernity criticism as
   well, so running both spends a second effort for the first benefit only. Keeping both
   is defensible if a reviewer specifically wants a C-band check.
2. **Does this reopen submission timing?** The manuscript is otherwise finished and
   de-telled, and this adds a results subsection and a figure, which means another review
   pass. That is a real cost against a paper that is ready to go, and the call is Hilawe's.
3. **What if the result is unflattering?** The pre-registered expectation above says
   agreement should drop. If it drops further than expected, or the contrast interval
   includes 1.0, the paper reports it. Confirm that is acceptable before the work starts,
   because the alternative is a comparison that only gets published when it is favourable,
   which is worse than not running it.
