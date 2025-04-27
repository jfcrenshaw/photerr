"""Parameter objects for the error model."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from typing import Any

import numpy as np

param_docstring = """
    Note for all the dictionary parameters below, you can pass a float
    instead and that value will be used for all bands. However at least
    one of these parameters must be a dictionary so that the model can
    infer the band names.

    Parameters
    ----------
    nYrObs : float
        Number of years of observations
    nVisYr : dict or float
        Mean number of visits per year in each band
    gamma : dict or float
        A band dependent parameter defined in Ivezic 2019
    m5 : dict or float
        A dictionary of single visit 5-sigma limiting magnitudes. For any
        bands for which you pass a value in m5, this will be the 5-sigma
        limiting magnitude used, and any values for that band in Cm, msky,
        theta, and km will be ignored.
    tvis : dict or float
        Exposure time in seconds for a single visit in each band.
    airmass : dict or float
        The airmass in each band.
    Cm : dict or float
        System contribution to the limiting magnitude in each band that is
        independent of observing conditions.
    dCmInf : dict or float
        The change in Cm in the limit of infinite visit time. This parameter
        only matters if tvis differs from tvisRef.
    msky : dict or float
        Zenith sky brightness in each band, in AB mag / arcsec^2.
    mskyDark : dict or float
        Zenith dark sky brightness in each band, in AB mag / arcsec^2.
    theta : dict or float
        Median zenith seeing FWHM in arcseconds for each band. This corresponds
        to seeingFwhmEff from OpSim runs.
    km : dict or float
        Atmospheric extinction at zenith in each band.
    tvisRef : float
        Reference exposure time used with tvis to scale read noise. This
        must correspond to the visit time used for the calculation of Cm.
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
    scale : dict; default=dict()
        A dictionary that rescales the error for the given bands. For example, if
        scale = {"u": 2}, then all the errors for the u band are doubled. This allows
        you to answer questions like "what happens to my science if the u band errors
        are doubled."
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
    validate : bool; True
        Whether or not to validate all the parameters.

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
    - tvis, airmass, Cm, dCmInf, msky, mskyDark, theta, and km, which are used
        to calculate the limiting magnitudes using Eq. 6 of Ivezic 2019 and
        Eq. 2-3 of Bianco 2022.

    Note if for any bands you pass a value in the m5 dictionary, the model will use
    that value for that band, regardless of the values in tvis, airmass, Cm, dCmInf,
    msky, mskyDark, theta, and km. However you must still provide airmass and theta
    if you wish to calculate errors for extended sources.

    When instantiating the ErrorParams object, it will determine which bands it has
    enough information to calculate errors for, and throw away all of the extraneous
    parameters. If you expect errors in a certain band, but the error model is not
    calculating errors for that band, check that you have provided all of the required
    parameters for that band.

    References
    ----------
    Ivezic 2019 - https://arxiv.org/abs/0805.2366
    Bianco 2022 - https://pstn-054.lsst.io
    van den Busch 2020 - http://arxiv.org/abs/2007.01846
    Kuijken 2019 - https://arxiv.org/abs/1902.11265
"""

# this dictionary defines the allowed types and values for every parameter
# we will use it for parameter validation during ErrorParams instantiation
_val_dict = {
    # param: ( is dict?, (allowed types), (allowed values), negative allowed? )
    "nYrObs": (False, (int, float), (), False),
    "nVisYr": (True, (int, float), (), False),
    "gamma": (True, (int, float), (), False),
    "m5": (True, (int, float), (), True),
    "tvis": (True, (int, float), (), False),
    "airmass": (True, (int, float), (), False),
    "Cm": (True, (int, float), (), True),
    "dCmInf": (True, (int, float), (), True),
    "msky": (True, (int, float), (), True),
    "mskyDark": (True, (int, float), (), True),
    "theta": (True, (int, float), (), False),
    "km": (True, (int, float), (), False),
    "tvisRef": (False, (int, float, type(None)), (), False),
    "sigmaSys": (False, (int, float), (), False),
    "sigLim": (False, (int, float), (), False),
    "ndMode": (False, (str,), ("flag", "sigLim"), None),
    "ndFlag": (False, (int, float), (), True),
    "absFlux": (False, (bool,), (), None),
    "extendedType": (False, (str,), ("point", "auto", "gaap"), None),
    "aMin": (False, (int, float), (), False),
    "aMax": (False, (int, float), (), False),
    "majorCol": (False, (str,), (), None),
    "minorCol": (False, (str,), (), None),
    "decorrelate": (False, (bool,), (), None),
    "highSNR": (False, (bool,), (), None),
    "scale": (True, (int, float), (), None),
    "errLoc": (False, (str,), ("after", "end", "alone"), None),
}


@dataclass
class ErrorParams:
    """Parameters for the photometric error models."""

    __doc__ += "\n" + param_docstring

    nYrObs: float
    nVisYr: dict[str, float] | float
    gamma: dict[str, float] | float

    m5: dict[str, float] | float = field(default_factory=lambda: {})

    tvis: dict[str, float] | float = field(default_factory=lambda: {})
    airmass: dict[str, float] | float = field(default_factory=lambda: {})
    Cm: dict[str, float] | float = field(default_factory=lambda: {})
    dCmInf: dict[str, float] | float = field(default_factory=lambda: {})
    msky: dict[str, float] | float = field(default_factory=lambda: {})
    mskyDark: dict[str, float] | float = field(default_factory=lambda: {})
    theta: dict[str, float] | float = field(default_factory=lambda: {})
    km: dict[str, float] | float = field(default_factory=lambda: {})

    tvisRef: float | None = None
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
    scale: dict[str, float] | float = field(default_factory=lambda: {})

    errLoc: str = "after"

    renameDict: InitVar[dict[str, str] | None] = None
    validate: InitVar[bool] = True

    def __post_init__(self, renameDict: dict[str, str] | None, validate: bool) -> None:
        """Rename bands and remove duplicate parameters."""
        # make sure all dictionaries are dictionaries
        self._convert_to_dict()

        # rename the bands
        self.rename_bands(renameDict)

        # validate the parameters
        if validate:
            self._validate_params()

        # clean up the dictionaries
        self._clean_dictionaries()

        # if using extended error types,
        if self.extendedType == "auto" or self.extendedType == "gaap":
            # make sure theta is provided for every band
            if set(self.theta.keys()) != set(self.nVisYr.keys()) or set(
                self.airmass.keys()
            ) != set(self.airmass.keys()):
                raise ValueError(
                    "If using one of the extended error types "
                    "(i.e. extendedType == 'auto' or 'gaap'), "
                    "then theta and airmass must be provided "
                    "for every band."
                )
            # make sure that aMin < aMax
            elif self.aMin > self.aMax:
                raise ValueError("aMin must be less than aMax.")

    def _convert_to_dict(self) -> None:
        """For dict parameters that aren't dicts, convert to dicts."""
        # first loop over all the parameters and get a list of every band
        bands = []
        for param in self.__dict__.values():
            if isinstance(param, dict):
                bands.extend(list(param.keys()))
        bands = list(set(bands))

        if len(bands) == 0:
            raise ValueError(
                "At least one of the dictionary parameters "
                "must actually be a dictionary."
            )

        # now loop over all the params and convert floats to dictionaries
        for key, (is_dict, *_) in _val_dict.items():
            # get the parameter saved under the key
            param = self.__dict__[key]

            # if it should be a dictionary but it's not
            if is_dict and not isinstance(param, dict):
                self.__dict__[key] = {band: param for band in bands}

    def _clean_dictionaries(self) -> None:
        """Remove unnecessary info from all of the dictionaries.

        This means that any bands explicitly listed in m5 will have their corresponding
        info removed from Cm, msky, theta, and km.

        Additionally, we will remove any band from all dictionaries that don't have
        an entry in all of nVisYr, gamma, and (m5 OR Cm + msky + theta + km).

        Finally, we will set scale=1 for all bands for which the scale isn't
        explicitly set.
        """
        # keep a running set of all the bands we will calculate errors for
        all_bands = set(self.nVisYr.keys()).intersection(self.gamma.keys())

        # remove the m5 bands from all other parameter dictionaries, and remove
        # bands from all_bands for which we don't have m5 data for
        ignore_dicts = ["m5", "nVisYr", "gamma", "scale"]
        for key, param in self.__dict__.items():
            # get the parameters that are dictionaries
            if isinstance(param, dict) and key not in ignore_dicts:
                if key != "theta" and key != "airmass":
                    # remove bands that we have explicit m5's for
                    self.__dict__[key] = {
                        band: val for band, val in param.items() if band not in self.m5
                    }

                # update the running list of bands that we have sufficient m5 data for
                all_bands = all_bands.intersection(
                    set(param.keys()).union(set(self.m5.keys()))
                )

        # remove all data for bands we will not be calculating errors for
        for key, param in self.__dict__.items():
            if isinstance(param, dict):
                self.__dict__[key] = {
                    band: val for band, val in param.items() if band in all_bands
                }

        # if there are no bands left, raise an error
        if len(all_bands) == 0:
            raise ValueError(
                "There are no bands left! You probably set one of the dictionary "
                "parameters (i.e. nVisYr, gamma, m5, Cm, dCmInf, msky, theta, or km) "
                "such that there was not enough info to calculate errors for any of "
                "the photometric bands. Remember that for each band, you must have an "
                "entry in nVisYr and gamma, plus either an entry in m5 or an entry "
                "in all of Cm, dCmInf, msky, theta, and km."
            )

        # finally, fill out the rest of the scale dictionary
        self.scale = {
            band: float(self.scale[band]) if band in self.scale else 1.0
            for band in self.nVisYr
        }

    def rename_bands(self, renameDict: dict[str, str] | None) -> None:
        """Rename the bands used in the error model.

        This method might be useful if you want to use the default parameters for an
        error model, but have given your bands different names.

        Parameters
        ----------
        renameDict: dict
            A dictionary containing key: value pairs {old_name: new_name}.
            For example map={"u": "lsst_u"} will rename the u band to lsst_u.
        """
        if renameDict is None:
            return
        if not isinstance(renameDict, dict):
            raise TypeError("renameDict must be a dict or None.")

        # loop over every parameter
        for key, param in self.__dict__.items():
            # get the parameters that are dictionaries
            if isinstance(param, dict):
                # rename bands in-place
                self.__dict__[key] = {
                    (
                        old_name if old_name not in renameDict else renameDict[old_name]
                    ): val
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

        # make sure that all of the keywords are in the class
        for key in kwargs:
            if key not in self.__dict__ and key not in ["renameDict", "validate"]:
                raise ValueError(
                    f"'{key}' is not a valid parameter name. "
                    "Please check the docstring."
                )

        # update parameters from keywords
        safe_copy = self.copy()
        try:
            # get the init variables
            renameDict = kwargs.pop("renameDict", None)
            validate = kwargs.pop("validate", True)

            # update all the other parameters
            for key, param in kwargs.items():
                setattr(self, key, param)

            # call post-init
            self.__post_init__(renameDict=renameDict, validate=validate)

        except Exception as error:
            self.__dict__ = safe_copy.__dict__
            raise error

    def copy(self) -> ErrorParams:
        """Return a full copy of this ErrorParams instance."""
        return deepcopy(self)

    @staticmethod
    def _check_single_param(
        key: str,
        subkey: str,
        param: Any,
        allowed_types: list,
        allowed_values: list,
        negative_allowed: bool,
    ) -> None:
        """Check that this single parameter has the correct type/value."""
        name = key if subkey is None else f"{key}.{subkey}"

        if not isinstance(param, allowed_types):
            raise TypeError(
                f"{name} is of type {type(param).__name__}, but should be "
                f"of type {', '.join(t.__name__ for t in allowed_types)}."
            )
        if len(allowed_values) > 0 and param not in allowed_values:
            raise ValueError(
                f"{name} has value {param}, but must be one of "
                f"{', '.join(v for v in allowed_values)}."
            )
        if (
            not negative_allowed
            and negative_allowed is not None
            and param is not None
            and param < 0
        ):
            raise ValueError(f"{name} has value {param}, but must be positive!")

    def _validate_params(self) -> None:
        """Validate parameter types and values."""
        # this dictionary defines the allowed types and values for every parameter

        # loop over parameter and check against the value dictionary
        for key, (
            is_dict,
            allowed_types,
            allowed_values,
            negative_allowed,
        ) in _val_dict.items():
            # get the parameter saved under the key
            param = self.__dict__[key]

            # do we have a dictionary on our hands?
            if isinstance(param, dict) and not is_dict:
                raise TypeError(f"{key} should not be a dictionary.")
            elif not isinstance(param, dict) and is_dict:
                raise TypeError(f"{key} should be a dictionary.")  # pragma: no cover
            if is_dict:
                # loop over contents and check types and values
                for subkey, subparam in param.items():
                    self._check_single_param(
                        key,
                        subkey,
                        subparam,
                        allowed_types,  # type: ignore
                        allowed_values,  # type: ignore
                        negative_allowed,  # type: ignore
                    )
            else:
                # check this single value
                self._check_single_param(
                    key,
                    None,  # type: ignore
                    param,
                    allowed_types,  # type: ignore
                    allowed_values,  # type: ignore
                    negative_allowed,  # type: ignore
                )
