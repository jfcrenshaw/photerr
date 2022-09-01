"""Parameter objects for the error model."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, Union

import numpy as np

param_docstring = """

    Parameters
    ----------
    nYrObs : float
        Number of years of observations
    nVisYr : dict
        Mean number of visits per year in each band
    gamma : dict
        A band dependent parameter defined in Ivezic 2019
    m5 : dict
        A dictionary of single visit 5-sigma limiting magnitudes. For any
        bands for which you pass a value in m5, this will be the 5-sigma
        limiting magnitude used, and any values for that band in Cm, msky,
        theta, and km will be ignored.
    tvis : float
        Exposure time in seconds for a single visit
    airmass : float
        The fiducial airmass
    Cm : dict
        A band dependent parameter defined in Ivezic 2019
    msky : dict
        Median zenith sky brightness in each band
    theta : dict
        Median zenith seeing FWHM in arcseconds for each band
    km : dict
        Atmospheric extinction in each band
    sigmaSys : float; default=0.005
        The irreducible error of the system in AB magnitudes. Sets the minimum
        photometric error.
    sigLim : float; default=0
        The n-sigma detection limit. Magnitudes beyond this limit are treated as
        non-detections. For example, if sigLim=1, then all magnitudes beyond the
        1-sigma limit in each band are treated as non-detections. sigLim=0
        corresponds to only treating negative fluxes as non-detections.
    ndMode : str; default="flag"
        The non-detection mode. I.e. how should the model handle non-detections?
        Non-detections are defined as magnitudes beyond sigLim (see above).
        There are two options:
            - "flag" - non-detections are flagged using the ndFlag (see below)
            - "sigLim" - magnitudes are clipped at the n-sigma limits. I.e. if
                sigLim=1 above, then all magnitudes greater than the 1-sigma limit
                in each band are replaced with the 1-sigma limiting magnitude.
    ndFlag : float; default=np.inf
        Flag for non-detections when ndMode == "flag".
    absFlux : bool; default=False
        Whether to take the absolute value of "observed" fluxes, before converting
        back to magnitudes. This removes the possibility of negative fluxes.
        absFlux=True together with sigLim=0 ensures that every galaxy is assigned
        an observed magnitude in every band, which is useful if you do not want to
        worry about non-detections.
    extendedType: str; default="point"
        Whether to use the error model for point sources or extended sources.
        For point sources, use "point". For extended sources, you can use "auto"
        or "gaap". See Notes below for more details on these models.
    aMin : float; default=2.0
        The minimum GAaP aperture diameter in arcseconds.
    aMax : float; default=0.7
        The maximum GAaP aperture diameter in arcseconds.
    majorCol : str; default="major"
        The name of the column containing the semi-major axes of the galaxies (in
        arcseconds). The length scales corresponds to the half-light radius.
    minorCol : str; default="minor"
        The name of the column containing the semi-minor axes of the galaxies (in
        arcseconds). The length scales corresponds to the half-light radius.
    decorrelate : bool; default=True
        Whether or not to decorrelate the photometric errors. If True, after calculating
        observed magnitudes, the errors are re-calculated using the observed magnitudes.
        If False, the original photometric errors are returned. Be warned, however,
        that in this case, the returned photometric errors are calculated from the true
        magnitudes, and thus provide a deterministic link back to the true magnitudes!
    highSNR : bool; default=False
        Whether to use the high SNR approximations given in Ivezic 2019. If False,
        then Eq. 5 from that paper is used to calculate (N/S)^2 in flux, and errors
        are Gaussian in flux space. If True, Eq. 5 is used to calculate the Gaussian
        variance in magnitude space.
    errLoc: str; default="after"
        Where to place the error columns in the output table. If "after", then the
        error columns will be placed immediately after the corresponding magnitude
        columns. If "end", then all of the error columns will be placed at the end
        of the table. If "alone", then the errors will be returned by themselves.
    renameDict : dict; optional
        A dictionary used to rename the bands in the parameters above that are
        dictionaries. This is useful if you want to use some of the default parameters
        but have given your bands different names. For example, if your bands are named
        "lsst_u", "lsst_g", etc., then you can provide
        renameDict={"u": "lsst_u", "g": "lsst_g", ...}, and all of the default
        parameters will be renamed. Additionally, if you are overriding any of the
        default dictionary parameters, you can provide those overrides using *either*
        the old or the new naming scheme.

    Notes
    -----
    Parameters for the error model from Ivezic 2019. We also implement a more general
    model that is also accurate in the low-SNR regime, and include errors for extended
    sources using the elliptical models from van den Busch 2020 (the "auto" model) and
    Kuijken 2019 (the "gaap" model). The "gaap" model identical to the "auto" model,
    except that it has minimum and maximum aperture sizes.

    When using the model for extended sources, you must provide the semi-major and
    semi-minor axes of the galaxies in arcseconds. These must be in columns whose
    names are given by majorCol and minorCol. In addition, you must provide the PSF
    size for each band in the theta dictionary (see above).

    Whether you are using the point source or extended source models, you must provide
    nVisYr and gamma for every band you wish to calculate errors for. In addition,
    you must provide either:
    - the single-visit 5-sigma limiting magnitude in the m5 dictionary
    - tvis and airmass, plus per-band parameters in the Cm, msky, theta, and km
        dictionaries, which are used to calculate the limiting magnitudes using
        Eq. 6 from Ivezic 2019.

    Note if for any bands you pass a value in the m5 dictionary, the model will use
    that value for that band, regardless of the values in Cm, msky, theta, and km.

    When instantiating the ErrorParams object, it will determine which bands it has
    enough information to calculate errors for, and throw away all of the extraneous
    parameters. If you expect errors in a certain band, but the error model is not
    calculating errors for that band, check that you have provided all of the required
    parameters for that band.

    References
    ----------
    Ivezic 2019 - https://arxiv.org/abs/0805.2366
    van den Busch 2020 - http://arxiv.org/abs/2007.01846
    Kuijken 2019 - https://arxiv.org/abs/1902.11265
"""


@dataclass
class ErrorParams:
    """Parameters for the photometric error models."""

    __doc__ += param_docstring

    nYrObs: float
    nVisYr: Dict[str, float]
    gamma: Dict[str, float]

    m5: Dict[str, float] = field(default_factory=lambda: {})

    tvis: float = None  # type: ignore
    airmass: float = None  # type: ignore
    Cm: Dict[str, float] = field(default_factory=lambda: {})
    msky: Dict[str, float] = field(default_factory=lambda: {})
    theta: Dict[str, float] = field(default_factory=lambda: {})
    km: Dict[str, float] = field(default_factory=lambda: {})

    sigmaSys: float = 0.005

    sigLim: float = 0
    ndMode: str = "flag"
    ndFlag: float = np.inf
    absFlux: bool = False

    extendedType: str = "point"
    aMin: float = 0.7
    aMax: float = 2.0
    majorCol: str = "major"
    minorCol: str = "minor"

    decorrelate: bool = True
    highSNR: bool = False
    errLoc: str = "after"

    renameDict: InitVar[Dict[str, str]] = None

    def __post_init__(self, renameDict: Union[Dict[str, str], None]) -> None:
        """Rename bands and remove duplicate parameters."""
        # if renameDict was provided, rename the bands
        if renameDict is not None:
            self.rename_bands(renameDict)

        # clean up the dictionaries
        self._clean_dictionaries()

        # if using extended error types, make sure theta is provided for every band
        if (
            self.extendedType == "auto" or self.extendedType == "gaap"
        ) and self.theta.keys() != self.nVisYr.keys():
            raise ValueError(
                "If using one of the extended error types "
                "(i.e. extendedType == 'auto' or 'gaap'), "
                "then theta must contain an entry for every band."
            )

    def _clean_dictionaries(self) -> None:
        """Remove unnecessary info from all of the dictionaries.

        This means that any bands explicitly listed in m5 will have their corresponding
        info removed from Cm, msky, theta, and km.

        Additionally, we will remove any band from all dictionaries that don't have
        an entry in all of nVisYr, gamma, and (m5 OR Cm + msky + theta + km).
        """
        # keep a running set of all the bands we will calculate errors for
        all_bands = set(self.nVisYr.keys()).intersection(self.gamma.keys())

        # remove the m5 bands from all other parameter dictionaries, and remove
        # bands from all_bands for which we don't have m5 data for
        ignore_dicts = ["m5", "nVisYr", "gamma", "theta"]
        for key, param in self.__dict__.items():
            # get the parameters that are dictionaries
            if isinstance(param, dict) and key not in ignore_dicts:
                # remove bands that we have explicit m5's for
                self.__dict__[key] = {
                    band: val for band, val in param.items() if band not in self.m5
                }

                # update the running list of bands that we have sufficient m5 data for
                all_bands = all_bands.intersection(
                    set(param.keys()).union(set(self.m5.keys()))
                )

        # finally, remove all of the data for bands that we will not be
        # calculating errors for
        for key, param in self.__dict__.items():
            if isinstance(param, dict):
                self.__dict__[key] = {
                    band: val for band, val in param.items() if band in all_bands
                }

        # if there are no bands left, raise an error
        if len(all_bands) == 0:
            raise ValueError(
                "There are no bands left! You probably set one of the dictionary "
                "parameters (i.e. nVisYr, gamma, m5, Cm, msky, theta, or km) such "
                "that there was not enough info to calculate errors for any of the "
                "photometric bands. Remember that for each band, you must have an "
                "entry in nVisYr and gamma, plus either an entry in m5 or an entry "
                "in all of Cm, msky, theta, and km."
            )

    def rename_bands(self, renameDict: Dict[str, str]) -> None:
        """Rename the bands used in the error model.

        This method might be useful if you want to use the default parameters for an
        error model, but have given your bands different names.

        Parameters
        ----------
        renameDict: dict
            A dictionary containing key: value pairs {old_name: new_name}.
            For example map={"u": "lsst_u"} will rename the u band to lsst_u.
        """
        # loop over every parameter
        for key, param in self.__dict__.items():
            # get the parameters that are dictionaries
            if isinstance(param, dict):
                # rename bands in-place
                self.__dict__[key] = {
                    old_name
                    if old_name not in renameDict
                    else renameDict[old_name]: val
                    for old_name, val in param.items()
                }

    def update(self, *args: dict, **kwargs: Any) -> None:
        """Update parameters using either a dictionary or keyword arguments."""
        # if non-keyword arguments passed, make sure it is just a single dictionary
        # and pass it back through as keyword arguments.
        if len(args) > 1:
            raise ValueError(
                "The only non-keyword argument that can be passed is a dictionary."
            )
        elif len(args) == 1:
            if not isinstance(args[0], dict):
                raise TypeError(
                    "The only non-keyword argument that can be passed is a dictionary."
                )
            else:
                self.update(**args[0], **kwargs)

        # update parameters from keywords
        safe_copy = self.copy()
        try:
            for key, param in kwargs.items():
                setattr(self, key, param)

            # call post-init
            self.__post_init__(renameDict=None)
        except ValueError as error:
            self.__dict__ = safe_copy.__dict__
            raise Warning("Aborting update!\n\n" + str(error))

    def copy(self) -> ErrorParams:
        """Return a full copy of this ErrorParams instance."""
        return deepcopy(self)
