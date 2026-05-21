"""Photometric error model for Roman."""

from dataclasses import dataclass, field

from photerr.model import ErrorModel, _make_survey_model
from photerr.params import ErrorParams, param_docstring

__all__ = [
    "RomanDeepErrorModel",
    "RomanDeepErrorParams",
    "RomanMediumErrorModel",
    "RomanMediumErrorParams",
    "RomanUltraDeepErrorModel",
    "RomanUltraDeepErrorParams",
    "RomanWideErrorModel",
    "RomanWideErrorParams",
]

# Module-level reference strings shared by all Roman params classes
_REF_GRAHAM = "    Graham 2020 - https://arxiv.org/abs/2004.07885"
_REF_ROTAC = (
    "\n    ROTAC Report - https://roman.gsfc.nasa.gov/"
    "science/ccs/ROTAC-Report-20250424-v1.pdf"
)
_REF_WFI = "\n    WFI Report - https://roman.gsfc.nasa.gov/science/WFI_technical.html"
_ROMAN_REFS = _REF_GRAHAM + _REF_ROTAC + _REF_WFI


@dataclass
class RomanWideErrorParams(ErrorParams):
    """Parameters for the Roman wide-field photometric error model.

    gamma taken from page 4 of Graham 2020.
    nYrObs and nVisYr set = 1, assuming that Roman is point-and-stare.

    Depth taken from appendix C.1 of the ROTAC report.

    PSF FWHMs taken from v1 of the WFI performance document.

    airmass=None for all bands: Roman is space-based, so no atmospheric
    PSF scaling is applied.
    """

    __doc__ += param_docstring
    __doc__ += _ROMAN_REFS

    nYrObs: float = 1.0
    nVisYr: dict[str, float] | float = 1.0
    gamma: dict[str, float] | float = 0.04

    m5: dict[str, float] | float = field(
        default_factory=lambda: {
            "H": 26.2,
        }
    )
    theta: dict[str, float] | float = field(
        default_factory=lambda: {
            "H": 0.128,
        }
    )
    # None signals space-based telescope: no airmass PSF scaling
    airmass: dict[str, float | None] | float | None = None


@dataclass
class RomanMediumErrorParams(RomanWideErrorParams):
    """Parameters for the Roman medium-field photometric error model.

    gamma taken from page 4 of Graham 2020.
    nYrObs and nVisYr set = 1, assuming that Roman is point-and-stare.

    Depth taken from appendix C.1 of the ROTAC report.

    PSF FWHMs taken from v1 of the WFI performance document.

    airmass=None for all bands: Roman is space-based, so no atmospheric
    PSF scaling is applied.
    """

    __doc__ += param_docstring
    __doc__ += _ROMAN_REFS

    m5: dict[str, float] | float = field(
        default_factory=lambda: {
            "Y": 26.5,
            "J": 26.4,
            "H": 26.4,
        }
    )
    theta: dict[str, float] | float = field(
        default_factory=lambda: {
            "Y": 0.087,
            "J": 0.106,
            "H": 0.128,
        }
    )
    airmass: dict[str, float | None] | float | None = None


@dataclass
class RomanDeepErrorParams(RomanWideErrorParams):
    """Parameters for the Roman deep-field photometric error model.

    gamma taken from page 4 of Graham 2020.
    nYrObs and nVisYr set = 1, assuming that Roman is point-and-stare.

    Depth taken from appendix C.1 of the ROTAC report.

    PSF FWHMs taken from v1 of the WFI performance document.

    airmass=None for all bands: Roman is space-based, so no atmospheric
    PSF scaling is applied.
    """

    __doc__ += param_docstring
    __doc__ += _ROMAN_REFS

    m5: dict[str, float] | float = field(
        default_factory=lambda: {
            "Z": 27.7,
            "Y": 27.7,
            "J": 27.6,
            "H": 27.5,
            "F": 27.0,
            "K": 25.9,
            "W": 28.3,
        }
    )
    theta: dict[str, float] | float = field(
        default_factory=lambda: {
            "Z": 0.073,
            "Y": 0.087,
            "J": 0.106,
            "H": 0.128,
            "F": 0.146,
            "K": 0.169,
            "W": 0.105,
        }
    )
    airmass: dict[str, float | None] | float | None = None


@dataclass
class RomanUltraDeepErrorParams(RomanWideErrorParams):
    """Parameters for the Roman ultra-deep-field photometric error model.

    gamma taken from page 4 of Graham 2020.
    nYrObs and nVisYr set = 1, assuming that Roman is point-and-stare.

    Depth taken from appendix C.1 of the ROTAC report.

    PSF FWHMs taken from v1 of the WFI performance document.

    airmass=None for all bands: Roman is space-based, so no atmospheric
    PSF scaling is applied.
    """

    __doc__ += param_docstring
    __doc__ += _ROMAN_REFS

    m5: dict[str, float] | float = field(
        default_factory=lambda: {
            "Y": 28.2,
            "J": 28.2,
            "H": 28.1,
        }
    )
    theta: dict[str, float] | float = field(
        default_factory=lambda: {
            "Y": 0.087,
            "J": 0.106,
            "H": 0.128,
        }
    )
    airmass: dict[str, float | None] | float | None = None


RomanWideErrorModel: type[ErrorModel] = _make_survey_model(RomanWideErrorParams)
RomanMediumErrorModel: type[ErrorModel] = _make_survey_model(RomanMediumErrorParams)
RomanDeepErrorModel: type[ErrorModel] = _make_survey_model(RomanDeepErrorParams)
RomanUltraDeepErrorModel: type[ErrorModel] = _make_survey_model(
    RomanUltraDeepErrorParams
)
