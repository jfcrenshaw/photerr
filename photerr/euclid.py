"""Photometric error model for Euclid."""

from dataclasses import dataclass, field

from photerr.model import ErrorModel, _make_survey_model
from photerr.params import ErrorParams, param_docstring

__all__ = [
    "EuclidDeepErrorModel",
    "EuclidDeepErrorParams",
    "EuclidWideErrorModel",
    "EuclidWideErrorParams",
]


@dataclass
class EuclidWideErrorParams(ErrorParams):
    """Parameters for the Euclid wide-field photometric error model.

    gamma taken from page 4 of Graham 2020.
    nYrObs and nVisYr set = 1, assuming that Euclid is point-and-stare.

    Depths taken from Table 7 of Scaramella 2021.

    PSF FWHM taken from Mellier 2024 (for IR bands this required
    multiplying stated FWHM in pixels by the NISP pixel size).

    airmass=None for all bands: Euclid is space-based, so no atmospheric
    PSF scaling is applied.
    """

    __doc__ += param_docstring
    __doc__ += "    Graham 2020 - https://arxiv.org/abs/2004.07885"
    __doc__ += "\n    Scaramella 2021 - https://arxiv.org/abs/2108.01201"
    __doc__ += "\n    Mellier 2024 - https://arxiv.org/abs/2405.13491"

    nYrObs: float = 1.0
    nVisYr: dict[str, float] | float = 1.0
    gamma: dict[str, float] | float = 0.04

    m5: dict[str, float] | float = field(
        default_factory=lambda: {
            "VIS": 26.2,
            "Y": 24.3,
            "J": 24.5,
            "H": 24.4,
        }
    )
    theta: dict[str, float] | float = field(
        default_factory=lambda: {
            "VIS": 0.13,
            "Y": 0.33,
            "J": 0.35,
            "H": 0.35,
        }
    )
    # None signals space-based telescope: no airmass PSF scaling
    airmass: dict[str, float | None] | float | None = None


@dataclass
class EuclidDeepErrorParams(EuclidWideErrorParams):
    """Parameters for the Euclid deep-field photometric error model.

    gamma taken from page 4 of Graham 2020.
    nYrObs and nVisYr set = 1, assuming that Euclid is point-and-stare.

    Depths taken from Table 7 of Scaramella 2021, adjusted for the SNR
    increases listed in Table 4 of Mellier 2024.

    PSF FWHM taken from Mellier 2024 (for IR bands this required
    multiplying stated FWHM in pixels by the NISP pixel size).

    airmass=None for all bands: Euclid is space-based, so no atmospheric
    PSF scaling is applied.
    """

    __doc__ += param_docstring
    __doc__ += "    Graham 2020 - https://arxiv.org/abs/2004.07885"
    __doc__ += "\n    Scaramella 2021 - https://arxiv.org/abs/2108.01201"
    __doc__ += "\n    Mellier 2024 - https://arxiv.org/abs/2405.13491"

    m5: dict[str, float] | float = field(
        default_factory=lambda: {
            "VIS": 27.9,
            "Y": 26.0,
            "J": 26.2,
            "H": 26.2,
        }
    )


EuclidWideErrorModel: type[ErrorModel] = _make_survey_model(EuclidWideErrorParams)
EuclidDeepErrorModel: type[ErrorModel] = _make_survey_model(EuclidDeepErrorParams)
