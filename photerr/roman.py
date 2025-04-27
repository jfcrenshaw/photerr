"""Photometric error model for Roman."""

from dataclasses import dataclass, field
from typing import Any

from photerr.model import ErrorModel
from photerr.params import ErrorParams, param_docstring


@dataclass
class RomanWideErrorParams(ErrorParams):
    """Parameters for the Roman wide-field photometric error model.

    gamma and limiting magnitudes taken from page 4 of Graham 2020.
    nYrObs and nVisYr set = 1, assuming that Roman is point-and-stare.

    Depth taken from appendix C.1 of the ROTAC report

    PSF FWHMs taken from v1 of the WFI performance document.
    """

    __doc__ += param_docstring
    __doc__ += "    Graham 2020 - https://arxiv.org/abs/2004.07885"
    __doc__ += (
        "\n    ROTAC Report - https://roman.gsfc.nasa.gov/"
        "science/ccs/ROTAC-Report-20250424-v1.pdf"
    )
    __doc__ += (
        "\n    WFI Report - https://roman.gsfc.nasa.gov/science/WFI_technical.html"
    )

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
    airmass: dict[str, float] | float = 0


class RomanWideErrorModel(ErrorModel):
    """Photometric error model for Roman wide field.

    Below is the parameter docstring:
    """

    __doc__ += RomanWideErrorParams.__doc__

    def __init__(self, **kwargs: Any) -> None:
        """Create a Roman wide-field error model.

        Keyword arguments override default values in RomanWideErrorParams.
        """
        super().__init__(RomanWideErrorParams(**kwargs))


@dataclass
class RomanMediumErrorParams(RomanWideErrorParams):
    """Parameters for the Roman medium-field photometric error model.

    gamma and limiting magnitudes taken from page 4 of Graham 2020.
    nYrObs and nVisYr set = 1, assuming that Roman is point-and-stare.

    Depth taken from appendix C.1 of the ROTAC report

    PSF FWHMs taken from v1 of the WFI performance document.
    """

    __doc__ += param_docstring
    __doc__ += "    Graham 2020 - https://arxiv.org/abs/2004.07885"
    __doc__ += (
        "\n    ROTAC Report - https://roman.gsfc.nasa.gov/"
        "science/ccs/ROTAC-Report-20250424-v1.pdf"
    )
    __doc__ += (
        "\n    WFI Report - https://roman.gsfc.nasa.gov/science/WFI_technical.html"
    )

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
    airmass: dict[str, float] | float = 0


class RomanMediumErrorModel(ErrorModel):
    """Photometric error model for Roman medium field.

    Below is the parameter docstring:
    """

    __doc__ += RomanMediumErrorParams.__doc__

    def __init__(self, **kwargs: Any) -> None:
        """Create a Roman medium-field error model.

        Keyword arguments override default values in RomanMediumErrorParams.
        """
        super().__init__(RomanMediumErrorParams(**kwargs))


@dataclass
class RomanDeepErrorParams(RomanWideErrorParams):
    """Parameters for the Roman deep-field photometric error model.

    gamma and limiting magnitudes taken from page 4 of Graham 2020.
    nYrObs and nVisYr set = 1, assuming that Roman is point-and-stare.

    Depth taken from appendix C.1 of the ROTAC report

    PSF FWHMs taken from v1 of the WFI performance document.
    """

    __doc__ += param_docstring
    __doc__ += "    Graham 2020 - https://arxiv.org/abs/2004.07885"
    __doc__ += (
        "\n    ROTAC Report - https://roman.gsfc.nasa.gov/"
        "science/ccs/ROTAC-Report-20250424-v1.pdf"
    )
    __doc__ += (
        "\n    WFI Report - https://roman.gsfc.nasa.gov/science/WFI_technical.html"
    )

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
    airmass: dict[str, float] | float = 0


class RomanDeepErrorModel(ErrorModel):
    """Photometric error model for Roman deep field.

    Below is the parameter docstring:
    """

    __doc__ += RomanDeepErrorParams.__doc__

    def __init__(self, **kwargs: Any) -> None:
        """Create a Roman deep-field error model.

        Keyword arguments override default values in RomanDeepErrorParams.
        """
        super().__init__(RomanDeepErrorParams(**kwargs))


@dataclass
class RomanUltraDeepErrorParams(RomanWideErrorParams):
    """Parameters for the Roman ultra-deep-field photometric error model.

    gamma and limiting magnitudes taken from page 4 of Graham 2020.
    nYrObs and nVisYr set = 1, assuming that Roman is point-and-stare.

    Depth taken from appendix C.1 of the ROTAC report

    PSF FWHMs taken from v1 of the WFI performance document.
    """

    __doc__ += param_docstring
    __doc__ += "    Graham 2020 - https://arxiv.org/abs/2004.07885"
    __doc__ += (
        "\n    ROTAC Report - https://roman.gsfc.nasa.gov/"
        "science/ccs/ROTAC-Report-20250424-v1.pdf"
    )
    __doc__ += (
        "\n    WFI Report - https://roman.gsfc.nasa.gov/science/WFI_technical.html"
    )

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
    airmass: dict[str, float] | float = 0


class RomanUltraDeepErrorModel(ErrorModel):
    """Photometric error model for Roman ultra-deep field.

    Below is the parameter docstring:
    """

    __doc__ += RomanUltraDeepErrorParams.__doc__

    def __init__(self, **kwargs: Any) -> None:
        """Create a Roman ultra-deep-field error model.

        Keyword arguments override default values in RomanUltraDeepErrorParams.
        """
        super().__init__(RomanUltraDeepErrorParams(**kwargs))
