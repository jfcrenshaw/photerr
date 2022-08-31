"""Photometric error model for LSST."""
from dataclasses import dataclass, field
from typing import Any, Dict

from photerr.model import PhotometricErrorModel
from photerr.params import ErrorParams, param_docstring


@dataclass
class LsstErrorParams(ErrorParams):
    """Parameters for the LSST photometric error model.

    Default values taken from pages 11, 12, 26 of Ivezic 2019.
    """

    __doc__ += param_docstring

    tvis: float = 30.0
    nYrObs: float = 10.0
    nVisYr: Dict[str, float] = field(
        default_factory=lambda: {
            "u": 5.6,
            "g": 8.0,
            "r": 18.4,
            "i": 18.4,
            "z": 16.0,
            "y": 16.0,
        }
    )
    gamma: Dict[str, float] = field(
        default_factory=lambda: {
            "u": 0.038,
            "g": 0.039,
            "r": 0.039,
            "i": 0.039,
            "z": 0.039,
            "y": 0.039,
        }
    )
    airmass: float = 1.2
    m5: Dict[str, float] = field(default_factory=lambda: {})
    Cm: Dict[str, float] = field(
        default_factory=lambda: {
            "u": 23.09,
            "g": 24.42,
            "r": 24.44,
            "i": 24.32,
            "z": 24.16,
            "y": 23.73,
        }
    )
    msky: Dict[str, float] = field(
        default_factory=lambda: {
            "u": 22.99,
            "g": 22.26,
            "r": 21.20,
            "i": 20.48,
            "z": 19.60,
            "y": 18.61,
        }
    )
    theta: Dict[str, float] = field(
        default_factory=lambda: {
            "u": 0.81,
            "g": 0.77,
            "r": 0.73,
            "i": 0.71,
            "z": 0.69,
            "y": 0.68,
        }
    )
    km: Dict[str, float] = field(
        default_factory=lambda: {
            "u": 0.491,
            "g": 0.213,
            "r": 0.126,
            "i": 0.096,
            "z": 0.069,
            "y": 0.170,
        }
    )


class LsstErrorModel(PhotometricErrorModel):
    """Photometric error model for Euclid."""

    def __init__(self, **kwargs: Any) -> None:
        """Create an LSST error model.

        Keyword arguments override default values in LsstErrorParams.
        """
        super().__init__(LsstErrorParams(), **kwargs)
