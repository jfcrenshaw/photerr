"""Tests for the error models.

The base ErrorModel object is only tested implicitly via the survey error models.

Need to write:
- test that increasing the number of years/observations per year/tvis reduce errors, etc
- test that increasing sky brightness/psf/airmass/sigmaSys decreases error, etc
- test the same for limiting mags
- test that m5 and non-coadded limiting mags match (might need to set sigmaSys=0)
- test that extended errors are greater than point errors
- test different flags for ndFlag
- test different sigLim stuff
- test absFlux (test that everything is finite)
- test decorrelate somehow
- test that highSNR is very close in the highSNR regime, but greater in lowSNR regime
- test different errLocs
"""
