"""Photometric error model for Roman."""

from dataclasses import dataclass, field
from typing import Any

from photerr.model import ErrorModel
from photerr.params import ErrorParams, param_docstring


@dataclass
class RomanErrorParams(ErrorParams):
    """Parameters for the Roman photometric error model.

    gamma and limiting magnitudes taken from page 4 of Graham 2020.
    nYrObs and nVisYr set = 1, assuming that Roman is point-and-stare.
    """

    __doc__ += param_docstring
    __doc__ += "    Graham 2020 - https://arxiv.org/abs/2004.07885"

    nYrObs: float = 1.0
    nVisYr: dict[str, float] | float = 1.0
    gamma: dict[str, float] | float = 0.04

    m5: dict[str, float] | float = field(
        default_factory=lambda: {
            "Y": 26.9,
            "J": 26.95,
            "H": 26.9,
            "F": 26.25,
        }
    )


class RomanErrorModel(ErrorModel):
    """Photometric error model for Roman.

    Below is the parameter docstring:
    """

    __doc__ += RomanErrorParams.__doc__

    def __init__(self, **kwargs: Any) -> None:
        """Create a Roman error model.

        Keyword arguments override default values in RomanErrorParams.
        """
        super().__init__(RomanErrorParams(**kwargs))
