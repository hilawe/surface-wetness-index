# Surface Wetness Index

[![tests](https://github.com/hilawe/surface-wetness-index/actions/workflows/ci.yml/badge.svg)](https://github.com/hilawe/surface-wetness-index/actions/workflows/ci.yml)

The Surface Wetness Index is an empirical passive-microwave detector of land
surface wetness, snow cover, and inundation, originally developed by Alan Basist
and colleagues at the NOAA National Climatic Data Center between roughly 1998
and 2004 from the Defense Meteorological Satellite Program (DMSP) Special Sensor
Microwave/Imager (SSM/I) record. From the seven SSM/I imager channels the
algorithm classifies each grid cell by surface and atmospheric condition through
a 42-node decision tree and retrieves three quantities: a 0 to 100 wetness index
(WET), an all-weather land surface skin temperature (RTEMP), and a snow and ice
flag derived from microwave scattering (SNOW). Through the 2000s the index was a
widely cited detector of surface wetness and inundation in the BAMS State of the
Climate reports and in agricultural and hydrological monitoring; development at
NCDC stopped after Basist left the agency.

This repository revives the algorithm on the modern SSM/I and Special Sensor
Microwave Imager/Sounder (SSMIS) record, on the Colorado State University (CSU)
Brightness Temperature Fundamental Climate Data Record, so the index can run
continuously from 1987 to the present on the same intercalibrated input that
NOAA stewards. The original C decision tree builds as a shared library that
serves as the reference oracle, and a vectorized NumPy port reproduces it cell
for cell at zero mismatches over fifteen million test cells across full branch
coverage. The product writers emit daily, weekly, and monthly Climate and
Forecast (CF) compliant NetCDF files, and the index has been validated against
four references that span the modern record, three physically independent and
one related microwave benchmark: ESA Climate Change
Initiative soil moisture, ERA5-Land soil moisture, U.S. Climate Reference
Network in-situ measurements, and Surface Water Microwave Product Series
inundation fraction.

## Principal Investigator

Hilawe Semunegus (NOAA National Centers for Environmental Information) is the
Principal Investigator for the project, responsible for the modern
reimplementation, validation, and stewardship of the rebuilt record. Alan Basist
is the original algorithm author and is credited as such in all derived
products. The algorithm itself stands in the NOAA passive-microwave hydrology
lineage of Norman Grody, Ralph Ferraro, and colleagues that produced the SSM/I
hydrological product family in the 1990s.

## What is here

- **C engine** (`src/`): the original 42-node Surface Wetness Index decision tree
  (`sig_recog.c`), wrapped as a shared library through `swi_api.c` so the
  Python port can call it through ctypes. The library is the reference oracle
  for the modern port and is built with `make` in `src/`.
- **NumPy port** (`python/swi/core_numpy.py`): a vectorized reimplementation
  that reproduces the C oracle bit for bit on the integer outputs and exactly
  on the float32 outputs across the full byte input domain.
- **Channel conventions** (`python/swi/channels.py`): the seven-channel byte
  packing (brightness temperature minus 70 K) inherited from the original
  daily one-third degree grids, and the output sentinels.
- **CSU FCDR gridded reader** (`python/swi/io_csu_grid.py`): direct ingest of
  the CSU Brightness Temperature Fundamental Climate Data Record daily 0.25
  degree gridded product, for SSM/I, SSMIS, and AMSR2 sensors.
- **85 to 91 GHz spectral calibration** (`python/swi/calib_8591.py`): the
  fitted correction that lets the algorithm, originally tuned at 85.5 GHz,
  run consistently on the SSMIS 91.655 GHz channel. The fit is derived from
  the F-15 and F-16 overlap in 2006.
- **Product writers** (`python/swi/product.py`, `python/swi/monthly.py`):
  daily, weekly, and monthly CF and Attribute Convention for Data Discovery
  compliant NetCDF outputs with honest provenance metadata.
- **Validation harness** (`python/swi/validate.py`, `python/swi/io_esacci.py`,
  `python/swi/io_era5.py`, `python/swi/io_uscrn.py`): co-location and skill
  metrics (rank correlation, anomaly correlation, detection contrast,
  categorical scores) against ESA CCI soil moisture, ERA5-Land, USCRN
  in-situ, and SWAMPS inundation.
- **Tests** (`python/tests/`): a pytest suite covering channel packing,
  calibration, grid geometry, NetCDF product invariants, validation
  metrics, and a deterministic bitwise regression of the NumPy port against
  the C oracle on randomized full-domain samples. Continuous integration
  runs the suite on every push.

## Running the tests

```bash
pip install -r requirements-test.txt
make -C src                  # build the C oracle (libswi.{so,dylib})
cd python && python -m pytest tests
```

Tests that need third-party reference data skip cleanly when those files are
not present, so the suite runs on any host with a C compiler and the
scientific Python stack. The regression test against the C oracle generates
its own synthetic input and always runs.

## Scientific and engineering highlights

- A faithful modern reimplementation of an empirical algorithm whose original
  source had drifted across two decades, validated to a zero-mismatch
  regression against the original C decision tree on fifteen million test
  cells with full branch coverage.
- A continuous 1987 to present input through the CSU Brightness Temperature
  FCDR, intercalibrated across the DMSP morning and late-morning constellation
  chains by NOAA-stewarded reprocessing.
- Validation against four reference datasets, three physically independent and
  one related microwave benchmark, shows the index is a
  strong wet versus dry surface-water detector, with detection contrast in the
  range expected from the original literature, rather than a quantitative soil
  moisture proxy. That framing matches what Basist and colleagues argued in
  the original 1998 and 2001 papers.
- The binary and NetCDF data contracts (channel order, byte packing, asc and
  dsc pass separation, CF metadata) are pinned by the test suite, not just
  documentation.

## Sibling project

Forward research that uses the same SSM/I and SSMIS brightness-temperature
record to derive a physically based land surface microwave emissivity, in
combination with the NOAA GridSat-B1 cloud-cleared infrared product, lives
in [github.com/hilawe/surface-wetness-lsme](https://github.com/hilawe/surface-wetness-lsme).
The two repositories share a minimal core of channel and grid conventions
but otherwise stand alone.

## References

The algorithm is grounded in these papers:

- Basist, A., N. C. Grody, T. C. Peterson, and C. N. Williams (1998).
  Using the Special Sensor Microwave Imager to monitor land surface
  temperatures, wetness, and snow cover.
  Journal of Applied Meteorology 37(9), 888 to 911.
- Basist, A., C. N. Williams, T. C. Peterson, N. Grody, P. Ross, and T. Karl
  (2001). Using the Special Sensor Microwave Imager to monitor land surface
  parameters.
  Journal of Hydrometeorology 2(3), 297 to 308.

The CSU Brightness Temperature FCDR used as input is documented in:

- Berg, W. and C. Kummerow (2018). Multisatellite intercalibrated CDR of
  brightness temperatures. Remote Sensing 10(8), 1306.

## License

This work was prepared by a U.S. Government employee as part of official duties
and is in the public domain in the United States (17 U.S.C. 105). To the extent
any rights exist, they are dedicated to the public domain under Creative Commons
Zero 1.0 (see `LICENSE`).
