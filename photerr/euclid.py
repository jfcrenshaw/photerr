"""Photometric error model for Euclid."""

from dataclasses import dataclass, field
from typing import Any

from photerr.model import ErrorModel
from photerr.params import ErrorParams, param_docstring


@dataclass
class EuclidErrorParams(ErrorParams):
    """Parameters for the Euclid photometric error model.

    gamma and limiting magnitudes taken from page 4 of Graham 2020.
    nYrObs and nVisYr set = 1, assuming that Euclid is point-and-stare.
    """

    __doc__ += param_docstring
    __doc__ += "    Graham 2020 - https://arxiv.org/abs/2004.07885"

    nYrObs: float = 1.0
    nVisYr: dict[str, float] | float = 1.0
    gamma: dict[str, float] | float = 0.04

    m5: dict[str, float] | float = field(
        default_factory=lambda: {
            "Y": 24.0,
            "J": 24.2,
            "H": 23.9,
        }
    )


class EuclidErrorModel(ErrorModel):
    """Photometric error model for Euclid.

    Below is the parameter docstring:
    """

    __doc__ += EuclidErrorParams.__doc__

    def __init__(self, **kwargs: Any) -> None:
        """Create a Euclid error model.

        Keyword arguments override default values in EuclidErrorParams.
        """
        super().__init__(EuclidErrorParams(**kwargs))
