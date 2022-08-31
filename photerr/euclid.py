"""Photometric error model for Euclid."""
from dataclasses import dataclass, field
from typing import Any, Dict

from photerr.model import PhotometricErrorModel
from photerr.params import ErrorParams, param_docstring


@dataclass
class EuclidErrorParams(ErrorParams):
    """Parameters for the Euclid photometric error model.

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
        }
    )
    gamma: Dict[str, float] = field(
        default_factory=lambda: {
            "Y": 0.04,
            "J": 0.04,
            "H": 0.04,
        }
    )
    m5: Dict[str, float] = field(
        default_factory=lambda: {
            "Y": 24.0,
            "J": 24.2,
            "H": 23.9,
        }
    )


class EuclidErrorModel(PhotometricErrorModel):
    """Photometric error model for Euclid."""

    def __init__(self, **kwargs: Any) -> None:
        """Create a Euclid error model.

        Keyword arguments override default values in EuclidErrorParams.
        """
        super().__init__(EuclidErrorParams(), **kwargs)
