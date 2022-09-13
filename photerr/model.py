"""The photometric error model."""
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd

from photerr.params import ErrorParams


class PhotometricErrorModel:
    """Base error model from Ivezic 2019.

    References
    ----------
    Ivezic 2019 - https://arxiv.org/abs/0805.2366
    van den Busch 2020 - http://arxiv.org/abs/2007.01846
    Kuijken 2019 - https://arxiv.org/abs/1902.11265
    """

    def __init__(self, *args: ErrorParams, **kwargs: Any) -> None:
        """Create an error model using the passed ErrorParams or keyword overrides.

        If you provide an ErrorParams object, those error parameters are used.
        If you provide keyword arguments, those values are used to override the defaults
        defined in the ErrorParams class.
        If you pass an ErrorParams object *and* keyword arguments, then the keyword
        values override the values in the passed ErrorParams object instead.
        """
        # check that non-keyword argument is just an ErrorParams object
        args_error_msg = (
            "The only non-keyword argument that can be passed "
            "is a single ErrorParams object."
        )
        if len(args) > 1:
            raise ValueError(args_error_msg)
        elif len(args) == 1 and not isinstance(args[0], ErrorParams):
            raise TypeError(args_error_msg)

        # assemble and save the final ErrorParams
        if len(args) == 0 and len(kwargs) == 0:
            raise ValueError(
                "Please provide either an ErrorParams object or the required keyword "
                "arguments from ErrorParams."
            )
        elif len(args) > 0 and len(kwargs) == 0:
            self._params = args[0]
        elif len(args) == 0 and len(kwargs) > 0:
            self._params = ErrorParams(**kwargs)
        else:
            self._params = args[0].copy()
            self._params.update(**kwargs)

        # make a list of the bands
        self._bands = [band for band in self._params.nVisYr.keys()]

        # calculate all of the 5-sigma limiting magnitudes
        self._calculate_m5()

        # calculate the limits for sigLim
        self._sigLim = self.getLimitingMags(self._params.sigLim, coadded=True)

    @property
    def params(self) -> ErrorParams:
        """The error model parameters in an ErrorParams objet."""  # noqa: D401
        return self._params

    def _calculate_m5(self) -> None:
        """Calculate the single-visit 5-sigma limiting magnitudes.

        Uses Eq. 6 from Ivezic 2019.
        However, if m5 has been explicitly passed for any bands, those values are
        used instead.
        """
        # calculate the m5 limiting magnitudes using Eq. 6
        m5 = {
            band: self.params.Cm[band]
            + 0.50 * (self.params.msky[band] - 21)
            + 2.5 * np.log10(0.7 / self.params.theta[band])
            + 1.25 * np.log10(self.params.tvis / 30)
            - self.params.km[band] * (self.params.airmass - 1)
            for band in self.params.Cm
        }
        m5.update(self.params.m5)

        self._all_m5 = {band: m5[band] for band in self._bands}

    def _get_area_ratio_auto(
        self, majors: np.ndarray, minors: np.ndarray, bands: list
    ) -> np.ndarray:
        """Get the ratio between PSF area and galaxy aperture area for "auto" model.

        Parameters
        ----------
        majors : np.ndarray
            The semi-major axes of the galaxies in arcseconds
        minors : np.ndarray
            The semi-minor axes of the galaxies in arcseconds
        bands : list
            The list of bands to calculate ratios for

        Returns
        -------
        np.ndarray
            The ratio of aperture size to PSF size for each band and galaxy.
        """
        # get the psf size for each band
        psf_size = np.array([self.params.theta[band] for band in bands])

        # convert PSF FWHM to Gaussian sigma
        psf_sig = psf_size / 2.355

        # calculate the area of the psf in each band
        A_psf = np.pi * psf_sig**2

        # calculate the area of the galaxy aperture in each band
        a_ap = np.sqrt(psf_sig[None, :] ** 2 + (2.5 * majors[:, None]) ** 2)
        b_ap = np.sqrt(psf_sig[None, :] ** 2 + (2.5 * minors[:, None]) ** 2)
        A_ap = np.pi * a_ap * b_ap

        # return their ratio
        return A_ap / A_psf

    def _get_area_ratio_gaap(
        self, majors: np.ndarray, minors: np.ndarray, bands: list
    ) -> np.ndarray:
        """Get the ratio between PSF area and galaxy aperture area for "gaap" model.

        Parameters
        ----------
        majors : np.ndarray
            The semi-major axes of the galaxies in arcseconds
        minors : np.ndarray
            The semi-minor axes of the galaxies in arcseconds
        bands : list
            The list of bands to calculate ratios for

        Returns
        -------
        np.ndarray
            The ratio of aperture size to PSF size for each band and galaxy.
        """
        # get the psf size for each band
        psf_size = np.array([self.params.theta[band] for band in bands])

        # convert PSF FWHM to Gaussian sigma
        psf_sig = psf_size / 2.355

        # convert min/max aperture diameter to Gaussian sigma
        aMin_sig = self.params.aMin / 2.355
        aMax_sig = self.params.aMax / 2.355

        # convert galaxy half-light radii to Gaussian sigmas
        # this conversion factor comes from half-IQR -> sigma
        majors /= 0.6745
        minors /= 0.6745

        # calculate the area of the psf in each band
        A_psf = np.pi * psf_sig**2

        # calculate the area of the galaxy aperture in each band
        a_ap = np.sqrt(psf_sig[None, :] ** 2 + majors[:, None] ** 2 + aMin_sig**2)
        a_ap[a_ap > aMax_sig] = aMax_sig
        b_ap = np.sqrt(psf_sig[None, :] ** 2 + minors[:, None] ** 2 + aMin_sig**2)
        b_ap[b_ap > aMax_sig] = aMax_sig
        A_ap = np.pi * a_ap * b_ap

        # return their ratio
        return A_ap / A_psf

    def _get_NSR(
        self, mags: np.ndarray, majors: np.ndarray, minors: np.ndarray, bands: list
    ) -> np.ndarray:
        """Calculate the noise-to-signal ratio.

        Uses Eqs 4 and 5 from Ivezic 2019.
        If using extended errors, also rescales with square root of area ratio.

        Parameters
        ----------
        mags : np.ndarray
            The magnitudes of the galaxies
        majors : np.ndarray
            The semi-major axes of the galaxies in arcseconds
        minors : np.ndarray
            The semi-minor axes of the galaxies in arcseconds
        bands : list
            The list of bands the galaxy is observed in

        Returns
        -------
        np.ndarray
            The ratio of aperture size to PSF size for each band and galaxy.
        """
        # get the 5-sigma limiting magnitudes for these bands
        m5 = np.array([self._all_m5[band] for band in bands])
        # and the values for gamma
        gamma = np.array([self.params.gamma[band] for band in bands])
        # and the number of visits per year
        nVisYr = np.array([self.params.nVisYr[band] for band in bands])

        # calculate x as defined in the paper
        x = 10 ** (0.4 * (mags - m5))

        # calculate the NSR for a single visit
        with np.errstate(invalid="ignore"):
            nsrRandSingleExp = np.sqrt((0.04 - gamma) * x + gamma * x**2)

        # calculate the NSR for the stacked image
        nStackedObs = nVisYr * self.params.nYrObs
        nsrRand = nsrRandSingleExp / np.sqrt(nStackedObs)

        # rescale according to the area ratio
        if self.params.extendedType == "auto":
            A_ratio = self._get_area_ratio_auto(majors, minors, bands)
        elif self.params.extendedType == "gaap":
            A_ratio = self._get_area_ratio_gaap(majors, minors, bands)
        else:
            A_ratio = 1  # type: ignore
        nsrRand *= np.sqrt(A_ratio)

        # get the irreducible system NSR
        if self.params.highSNR:
            nsrSys = self.params.sigmaSys
        else:
            nsrSys = 10 ** (self.params.sigmaSys / 2.5) - 1

        # calculate the total NSR
        nsr = np.sqrt(nsrRand**2 + nsrSys**2)

        return nsr

    def _get_obs_and_errs(
        self,
        mags: np.ndarray,
        majors: np.ndarray,
        minors: np.ndarray,
        bands: list,
        sigLim: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the noise-to-signal ratio.

        Uses Eqs 4 and 5 from Ivezic 2019.
        If using extended errors, also rescales with square root of area ratio.

        Parameters
        ----------
        mags : np.ndarray
            The magnitudes of the galaxies
        majors : np.ndarray
            The semi-major axes of the galaxies in arcseconds
        minors : np.ndarray
            The semi-minor axes of the galaxies in arcseconds
        bands : list
            The list of bands the galaxy is observed in
        sigLim : np.ndarray
            The n-sigma limits for non-detection
        rng : np.random.Generator
            A numpy random number generator

        Returns
        -------
        np.ndarray
            The ratio of aperture size to PSF size for each band and galaxy.
        """
        # get the NSR for all galaxies
        nsr = self._get_NSR(mags, majors, minors, bands)

        if self.params.highSNR:
            # in the high SNR approximation, mag err ~ nsr, and we can model
            # errors as Gaussian in magnitude space

            # calculate observed magnitudes
            obsMags = rng.normal(loc=mags, scale=nsr)

            # if ndMode == sigLim, then clip all magnitudes at the n-sigma limit
            if self.params.ndMode == "sigLim":
                obsMags = np.clip(obsMags, None, sigLim)

            # if decorrelate, then we calculate new errors using the observed mags
            if self.params.decorrelate:
                nsr = self._get_NSR(obsMags, majors, minors, bands)

            obsMagErrs = nsr

        else:
            # in the more accurate error model, we acknowledge err != nsr,
            # and we model errors as Gaussian in flux space

            # calculate observed magnitudes
            fluxes = 10 ** (mags / -2.5)
            obsFluxes = fluxes * (1 + rng.normal(scale=nsr))
            if self.params.absFlux:
                obsFluxes = np.abs(obsFluxes)
            with np.errstate(divide="ignore"):
                obsMags = -2.5 * np.log10(np.clip(obsFluxes, 0, None))

            # if ndMode == sigLim, then clip all magnitudes at the n-sigma limit
            if self.params.ndMode == "sigLim":
                obsMags = np.clip(obsMags, None, sigLim)

            # if decorrelate, then we calculate new errors using the observed mags
            if self.params.decorrelate:
                nsr = self._get_NSR(obsMags, majors, minors, bands)

            obsMagErrs = 2.5 * np.log10(1 + nsr)

        return obsMags, obsMagErrs

    def __call__(
        self, catalog: pd.DataFrame, random_state: Union[np.random.Generator, int]
    ) -> pd.DataFrame:
        """Calculate photometric errors for the catalog and return in DataFrame.

        Parameters
        ----------
        catalog : pd.DataFrame
            The input catalog of galaxies in a pandas DataFrame.
        random_sate : np.random.Generator or int
            The random state. Can either be a numpy random generator
            (e.g. np.random.default_rng(42)), or an integer, which is then used
            to seed np.random.default_rng.

        Returns
        -------
        pd.DataFrame
        """
        # set the rng
        if isinstance(random_state, np.random.Generator):
            rng = random_state
        elif isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            raise TypeError("random_state must be a numpy random generator or an int.")

        # get the bands we will calculate errors for
        bands = [band for band in catalog.columns if band in self._bands]

        # calculate the n-sigma limits
        sigLim = np.array([self._sigLim[band] for band in bands])

        # get the numpy array of magnitudes
        mags = catalog[bands].to_numpy()

        # get the semi-major and semi-minor axes
        if self.params.extendedType == "auto" or self.params.extendedType == "gaap":
            majors = catalog[self.params.majorCol].to_numpy()
            minors = catalog[self.params.minorCol].to_numpy()
        else:
            majors = None
            minors = None

        # get observed magnitudes and errors
        obsMags, obsMagErrs = self._get_obs_and_errs(
            mags, majors, minors, bands, sigLim, rng
        )

        # flag all non-detections with the ndFlag
        if self.params.ndMode == "flag":
            idx = (~np.isfinite(obsMags)) | (obsMags > sigLim)
            obsMags[idx] = self.params.ndFlag
            obsMagErrs[idx] = self.params.ndFlag

        # save the observations in a DataFrame
        errDf = pd.DataFrame(
            obsMagErrs, columns=[f"{band}_err" for band in bands], index=catalog.index
        )
        if self.params.errLoc == "alone":
            obsCatalog = errDf
        else:
            magDf = catalog.copy()
            magDf[bands] = obsMags
            obsCatalog = pd.concat([magDf, errDf], axis=1)

        if self.params.errLoc == "after":
            # reorder the columns so that the error columns come right after the
            # respective magnitude columns
            columns = catalog.columns.tolist()
            for band in bands:
                columns.insert(columns.index(band) + 1, f"{band}_err")
            obsCatalog = obsCatalog[columns]

        return obsCatalog

    def getLimitingMags(self, nSigma: float = 5, coadded: bool = True) -> dict:
        """Return the limiting magnitudes for point sources in all bands.

        This method essentially reverse engineers the _get_NSR method so that we
        calculate what magnitude results in NSR = 1/nSigma.
        (NSR is noise-to-signal ratio; NSR = 1/SNR)

        Parameters
        ----------
        nSigma : float; default=5
            Sets which limiting magnitude to return. E.g. nSigma=1 returns the 1-sigma
            limiting magnitude. In other words, nSigma is equal to the signal-to-noise
            ratio (SNR) of the limiting magnitudes.
        coadded : bool; default=True
            If True, returns the limiting magnitudes for a coadded image. If False,
            returns the limiting magnitudes for a single visit.

        Returns
        -------
        dict
            The dictionary of limiting magnitudes
        """
        bands = self._bands

        # if nSigma is zero, return infinite magnitude limits
        if np.isclose(0, nSigma):
            return {band: np.inf for band in bands}

        # get the 5-sigma limiting magnitudes for these bands
        m5 = np.array([self._all_m5[band] for band in bands])
        # and the values for gamma
        gamma = np.array([self.params.gamma[band] for band in bands])
        # and the number of visits per year
        nVisYr = np.array([self.params.nVisYr[band] for band in bands])

        # get the number of exposures
        if coadded:
            nStackedObs = nVisYr * self.params.nYrObs
        else:
            nStackedObs = 1  # type: ignore

        # get the irreducible system error
        if self.params.highSNR:
            nsrSys = self.params.sigmaSys
        else:
            nsrSys = 10 ** (self.params.sigmaSys / 2.5) - 1

        # calculate the random NSR that a single exposure must have
        nsrRandSingleExp = np.sqrt((1 / nSigma**2 - nsrSys**2) * nStackedObs)

        # calculate the value of x that corresponds to this NSR
        # this is just the quadratic equation applied to Eq 5 from Ivezic 2019
        x = (
            (gamma - 0.04)
            + np.sqrt((gamma - 0.04) ** 2 + 4 * gamma * nsrRandSingleExp**2)
        ) / (2 * gamma)

        # convert x to a limiting magnitude
        limiting_mags = m5 + 2.5 * np.log10(x)

        # return as a dictionary
        return dict(zip(bands, limiting_mags))

    def __repr__(self) -> str:  # noqa: D105
        return "Photometric error model with parameters:\n\n" + str(self.params)
