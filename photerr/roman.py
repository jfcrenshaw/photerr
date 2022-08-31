"""Photometric error model for Roman."""
from dataclasses import dataclass, field
from typing import Any, Dict

from photerr.model import PhotometricErrorModel
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
    nVisYr: Dict[str, float] = field(
        default_factory=lambda: {
            "Y": 1.0,
            "J": 1.0,
            "H": 1.0,
            "F": 1.0,
        }
    )
    gamma: Dict[str, float] = field(
        default_factory=lambda: {
            "Y": 0.04,
            "J": 0.04,
            "H": 0.04,
            "F": 0.04,
        }
    )
    m5: Dict[str, float] = field(
        default_factory=lambda: {
            "Y": 26.9,
            "J": 26.95,
            "H": 26.9,
            "F": 26.25,
        }
    )


class RomanErrorModel(PhotometricErrorModel):
    """Photometric error model for Roman."""

    def __init__(self, **kwargs: Any) -> None:
        """Create a Roman error model.

        Keyword arguments override default values in RomanErrorParams.
        """
        super().__init__(RomanErrorParams(), **kwargs)
