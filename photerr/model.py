"""The photometric error model."""

from typing import Any

import numpy as np
import pandas as pd

from photerr.params import ErrorParams


class ErrorModel:
    """Base error model from Ivezic 2019.

    Below is the parameter docstring:
    """

    __doc__ += ErrorParams.__doc__

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
        self._bands = list(self._params.nVisYr.keys())

        # calculate all of the 5-sigma limiting magnitudes
        self._calculate_m5()

    @property
    def params(self) -> ErrorParams:
        """The error model parameters in an ErrorParams objet."""  # noqa: D401
        return self._params

    def _calculate_m5(self) -> None:
        """Calculate the single-visit 5-sigma limiting magnitudes.

        Uses Eq. 6 from Ivezic 2019 and Eq. 2-3 of Bianco 2022.
        However, if m5 has been explicitly passed for any bands,
        those values are used instead.
        """
        # calculate m5 for the bands with the requisite info
        m5 = {}
        for band in self.params.Cm:
            # calculate the scaled visit time using Eq. 3 of Bianco 2022
            tau = (
                self.params.tvis[band]
                / self.params.tvisRef
                * 10 ** ((self.params.msky[band] - self.params.mskyDark[band]) / -2.5)
            )

            # calculate Cm exposure time adjustment using Eq. 2 from Bianco 2022
            dCmTau = self.params.dCmInf[band] - 1.25 * np.log10(
                1 + (10 ** (0.8 * self.params.dCmInf[band]) - 1) / tau
            )

            # calculate m5 using Eq. 6 from Ivezic 2019 and Eq. 2 from Bianco 2022
            m5[band] = (
                self.params.Cm[band]
                + 0.50 * (self.params.msky[band] - 21)
                + 2.5 * np.log10(0.7 / self.params.theta[band])
                + 1.25 * np.log10(self.params.tvis[band] / 30)
                - self.params.km[band] * (self.params.airmass[band] - 1)
                + dCmTau
            )

        # add the explicitly passed m5's to our dictionary
        m5.update(self.params.m5)

        # save all the single-visit m5's together
        self._all_m5 = {band: m5[band] for band in self._bands}

    def _get_area_ratio_auto(
        self,
        majors: np.ndarray,
        minors: np.ndarray,
        bands: list,
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
        airmass = np.array([self.params.airmass[band] for band in bands])
        for i in range(len(airmass)):
            if airmass[i] > 0:
                psf_size[i] *= airmass[i] ** 0.6

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
        self,
        majors: np.ndarray,
        minors: np.ndarray,
        bands: list,
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
        airmass = np.array([self.params.airmass[band] for band in bands])
        for i in range(len(airmass)):
            if airmass[i] > 0:
                psf_size[i] *= airmass[i] ** 0.6

        # convert PSF FWHM to Gaussian sigma
        psf_sig = psf_size / 2.355

        # convert min/max aperture diameter to Gaussian sigma
        aMin_sig = self.params.aMin / 2.355
        aMax_sig = self.params.aMax / 2.355

        # convert galaxy half-light radii to Gaussian sigmas
        # this conversion factor comes from half-IQR -> sigma
        majors = majors / 0.6745
        minors = minors / 0.6745

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

    def _get_nsr_from_mags(
        self,
        mags: np.ndarray,
        majors: np.ndarray,
        minors: np.ndarray,
        bands: list,
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
            The noise-to-signal ratio of each galaxy
        """
        # get the 5-sigma limiting magnitudes for these bands
        m5 = np.array([self._all_m5[band] for band in bands])
        # and the values for gamma
        gamma = np.array([self.params.gamma[band] for band in bands])
        # and the number of visits per year
        nVisYr = np.array([self.params.nVisYr[band] for band in bands])
        # and the scales
        scale = np.array([self.params.scale[band] for band in bands])

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

        # rescale the NSR
        nsrRand *= scale

        # get the irreducible system NSR
        if self.params.highSNR:
            nsrSys = self.params.sigmaSys
        else:
            nsrSys = 10 ** (self.params.sigmaSys / 2.5) - 1

        # calculate the total NSR
        nsr = np.sqrt(nsrRand**2 + nsrSys**2)

        return nsr

    def _get_mags_from_nsr(
        self,
        nsr: np.ndarray,
        majors: np.ndarray,
        minors: np.ndarray,
        bands: list,
        coadded: bool = True,
    ) -> np.ndarray:
        """Calculate magnitudes that correspond to the given NSRs.

        Essentially inverts self._get_nsr_from_mags().

        Parameters
        ----------
        nsr : np.ndarray
            The noise-to-signal ratios of the galaxies
        majors : np.ndarray
            The semi-major axes of the galaxies in arcseconds
        minors : np.ndarray
            The semi-minor axes of the galaxies in arcseconds
        bands : list
            The list of bands the galaxy is observed in
        coadded : bool; default=True
            If True, assumes NSR is after coaddition.

        Returns
        -------
        np.ndarray
            The magnitude corresponding to the NSR for each galaxy
        """
        # get the 5-sigma limiting magnitudes for these bands
        m5 = np.array([self._all_m5[band] for band in bands])
        # and the values for gamma
        gamma = np.array([self.params.gamma[band] for band in bands])
        # and the number of visits per year
        nVisYr = np.array([self.params.nVisYr[band] for band in bands])
        # and the scales
        scale = np.array([self.params.scale[band] for band in bands])

        # get the irreducible system NSR
        if self.params.highSNR:
            nsrSys = self.params.sigmaSys
        else:
            nsrSys = 10 ** (self.params.sigmaSys / 2.5) - 1

        # calculate the random NSR
        nsrRand = np.sqrt(nsr**2 - nsrSys**2)

        # rescale the NSR
        nsrRand /= scale

        # rescale according to the area ratio
        if majors is not None and minors is not None:
            if self.params.extendedType == "auto":
                A_ratio = self._get_area_ratio_auto(majors, minors, bands)
            elif self.params.extendedType == "gaap":
                A_ratio = self._get_area_ratio_gaap(majors, minors, bands)
            else:
                A_ratio = 1  # type: ignore
            nsrRand = nsrRand / np.sqrt(A_ratio)

        # get the number of exposures
        if coadded:
            nStackedObs = nVisYr * self.params.nYrObs
        else:
            nStackedObs = 1  # type: ignore

        # calculate the NSR for a single image
        nsrRandSingleExp = nsrRand * np.sqrt(nStackedObs)

        # calculate the value of x that corresponds to this NSR
        # this is just the quadratic equation applied to Eq 5 from Ivezic 2019
        x = (
            (gamma - 0.04)
            + np.sqrt((gamma - 0.04) ** 2 + 4 * gamma * nsrRandSingleExp**2)
        ) / (2 * gamma)

        # convert x to magnitudes
        mags = m5 + 2.5 * np.log10(x)

        return mags

    def _get_obs_and_errs(
        self,
        mags: np.ndarray,
        majors: np.ndarray,
        minors: np.ndarray,
        bands: list,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
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
        rng : np.random.Generator
            A numpy random number generator

        Returns
        -------
        np.ndarray
            The ratio of aperture size to PSF size for each band and galaxy.
        """
        # get the NSR for all galaxies
        nsr = self._get_nsr_from_mags(mags, majors, minors, bands)

        if self.params.highSNR:
            # in the high SNR approximation, mag err ~ nsr, and we can model
            # errors as Gaussian in magnitude space

            # calculate observed magnitudes
            obsMags = rng.normal(loc=mags, scale=nsr)

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

        # if decorrelate, then we calculate new errors using the observed mags
        if self.params.decorrelate:
            nsr = self._get_nsr_from_mags(obsMags, majors, minors, bands)

        # if ndMode == sigLim, then clip at the n-sigma limit
        if self.params.ndMode == "sigLim":
            # determine the nsr limit
            with np.errstate(divide="ignore"):
                nsrLim = np.divide(1, self.params.sigLim)

            # calculate limiting magnitudes for each galaxy
            magLim = self._get_mags_from_nsr(nsrLim, majors, minors, bands)

            # clip mags and nsr's at this limit
            nsr = np.clip(nsr, 0, nsrLim)
            obsMags = np.clip(obsMags, None, magLim)

        if self.params.highSNR:
            obsMagErrs = nsr
        else:
            obsMagErrs = 2.5 * np.log10(1 + nsr)

        return obsMags, obsMagErrs

    def __call__(
        self,
        catalog: pd.DataFrame,
        random_state: np.random.Generator | int | None = None,
    ) -> pd.DataFrame:
        """Calculate photometric errors for the catalog and return in DataFrame.

        Parameters
        ----------
        catalog : pd.DataFrame
            The input catalog of galaxies in a pandas DataFrame.
        random_sate : np.random.Generator, int, or None
            The random state. Can either be a numpy random generator
            (e.g. np.random.default_rng(42)), an integer (which is used
            to seed np.random.default_rng), or None.

        Returns
        -------
        pd.DataFrame
        """
        # set the rng
        rng = np.random.default_rng(random_state)

        # get the bands we will calculate errors for
        bands = [band for band in catalog.columns if band in self._bands]

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
        obsMags, obsMagErrs = self._get_obs_and_errs(mags, majors, minors, bands, rng)

        # flag all non-detections with the ndFlag
        if self.params.ndMode == "flag":
            # calculate SNR
            if self.params.highSNR:
                snr = 1 / obsMagErrs
            else:
                snr = 1 / (10 ** (obsMagErrs / 2.5) - 1)

            # flag non-finite mags and where SNR is below sigLim
            idx = (~np.isfinite(obsMags)) | (snr < self.params.sigLim)
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

    def getLimitingMags(
        self,
        nSigma: float = 5,
        coadded: bool = True,
        aperture: float = 0,
    ) -> dict:
        """Return the limiting magnitudes in all bands.

        Parameters
        ----------
        nSigma : float; default=5
            Sets which limiting magnitude to return. E.g. nSigma=1 returns the 1-sigma
            limiting magnitude. In other words, nSigma is equal to the signal-to-noise
            ratio (SNR) of the limiting magnitudes.
        coadded : bool; default=True
            If True, returns the limiting magnitudes for a coadded image. If False,
            returns the limiting magnitudes for a single visit.
        aperture : float, default=0
            Radius of circular aperture in arcseconds. If zero, limiting magnitudes
            are for point sources. If greater than zero, limiting magnitudes are
            for objects of that size. Increasing the aperture decreases the
            signal-to-noise ratio. This only has an impact if extendedType != "point.

        Returns
        -------
        dict
            The dictionary of limiting magnitudes
        """
        # return as a dictionary
        return dict(
            zip(
                self._bands,
                self._get_mags_from_nsr(
                    1 / nSigma,  # type: ignore
                    np.array([aperture]),
                    np.array([aperture]),
                    self._bands,
                    coadded,
                ).flatten(),
            )
        )

    def __repr__(self) -> str:  # pragma: no cover
        """Print the error model parameters."""
        return "Photometric error model with parameters:\n\n" + str(self.params)
