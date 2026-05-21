"""Photometric error model for LSST using defaults from Ivezic 2019."""

from dataclasses import dataclass, field

from photerr.model import ErrorModel, _make_survey_model
from photerr.params import ErrorParams, param_docstring

__all__ = ["LsstErrorModelV1", "LsstErrorParamsV1"]


@dataclass
class LsstErrorParamsV1(ErrorParams):
    """Parameters for the LSST photometric error model.

    Default values taken from pages 11, 12, 26 of Ivezic 2019.
    """

    __doc__ += param_docstring

    nYrObs: float = 10.0
    nVisYr: dict[str, float] | float = field(
        default_factory=lambda: {
            "u": 5.6,
            "g": 8.0,
            "r": 18.4,
            "i": 18.4,
            "z": 16.0,
            "y": 16.0,
        }
    )
    gamma: dict[str, float] | float = field(
        default_factory=lambda: {
            "u": 0.038,
            "g": 0.039,
            "r": 0.039,
            "i": 0.039,
            "z": 0.039,
            "y": 0.039,
        }
    )

    m5: dict[str, float] | float = field(default_factory=lambda: {})

    tvis: dict[str, float] | float = 30
    airmass: dict[str, float | None] | float | None = 1.2
    Cm: dict[str, float] | float = field(
        default_factory=lambda: {
            "u": 23.09,
            "g": 24.42,
            "r": 24.44,
            "i": 24.32,
            "z": 24.16,
            "y": 23.73,
        }
    )
    dCmInf: dict[str, float] | float = field(
        default_factory=lambda: {
            "u": 0.62,
            "g": 0.18,
            "r": 0.10,
            "i": 0.07,
            "z": 0.05,
            "y": 0.04,
        }
    )
    msky: dict[str, float] | float = field(
        default_factory=lambda: {
            "u": 22.99,
            "g": 22.26,
            "r": 21.20,
            "i": 20.48,
            "z": 19.60,
            "y": 18.61,
        }
    )
    # mskyDark equals msky in this dataset: Ivezic 2019 uses a single sky brightness
    # value, so the reference and actual sky brightness are the same.
    mskyDark: dict[str, float] | float = field(
        default_factory=lambda: {
            "u": 22.99,
            "g": 22.26,
            "r": 21.20,
            "i": 20.48,
            "z": 19.60,
            "y": 18.61,
        }
    )
    theta: dict[str, float] | float = field(
        default_factory=lambda: {
            "u": 0.92,
            "g": 0.87,
            "r": 0.83,
            "i": 0.80,
            "z": 0.78,
            "y": 0.76,
        }
    )
    km: dict[str, float] | float = field(
        default_factory=lambda: {
            "u": 0.491,
            "g": 0.213,
            "r": 0.126,
            "i": 0.096,
            "z": 0.069,
            "y": 0.170,
        }
    )

    tvisRef: float | None = 30


LsstErrorModelV1: type[ErrorModel] = _make_survey_model(LsstErrorParamsV1)
