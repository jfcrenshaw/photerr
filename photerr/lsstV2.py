"""Photometric error model for LSST as of Feb 28, 2024."""

from dataclasses import dataclass, field
from typing import Any

from photerr.model import ErrorModel
from photerr.params import ErrorParams, param_docstring


@dataclass
class LsstErrorParamsV2(ErrorParams):
    """Parameters for the LSST photometric error model.

    Default values match v1.9 of the official LSST throughputs, including the
    silver coatings on all three mirrors. The values for tvis and nVisYr were
    taken from one of the OpSim runs that Lynne Jones said is likely to resemble
    the final exposure times and visit distributions.
    """

    __doc__ += param_docstring

    nYrObs: float = 10.0
    nVisYr: dict[str, float] | float = field(
        default_factory=lambda: {
            "u": 6.3,
            "g": 6.8,
            "r": 18.1,
            "i": 18.8,
            "z": 16.6,
            "y": 16.3,
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

    tvis: dict[str, float] | float = field(
        default_factory=lambda: {
            "u": 38,
            "g": 30,
            "r": 30,
            "i": 30,
            "z": 30,
            "y": 30,
        }
    )
    airmass: dict[str, float] | float = field(
        default_factory=lambda: {
            "u": 1.15,
            "g": 1.15,
            "r": 1.14,
            "i": 1.15,
            "z": 1.16,
            "y": 1.16,
        }
    )
    Cm: dict[str, float] | float = field(
        default_factory=lambda: {
            "u": 22.968,
            "g": 24.582,
            "r": 24.602,
            "i": 24.541,
            "z": 24.371,
            "y": 23.840,
        }
    )
    dCmInf: dict[str, float] | float = field(
        default_factory=lambda: {
            "u": 0.543,
            "g": 0.088,
            "r": 0.043,
            "i": 0.028,
            "z": 0.019,
            "y": 0.016,
        }
    )
    msky: dict[str, float] | float = field(
        default_factory=lambda: {
            "u": 22.65,
            "g": 22.07,
            "r": 21.10,
            "i": 19.99,
            "z": 19.04,
            "y": 18.28,
        }
    )
    mskyDark: dict[str, float] | float = field(
        default_factory=lambda: {
            "u": 23.052,
            "g": 22.254,
            "r": 21.198,
            "i": 20.463,
            "z": 19.606,
            "y": 18.602,
        }
    )
    theta: dict[str, float] | float = field(
        default_factory=lambda: {
            "u": 1.22,
            "g": 1.09,
            "r": 1.02,
            "i": 0.99,
            "z": 1.01,
            "y": 0.99,
        }
    )
    km: dict[str, float] | float = field(
        default_factory=lambda: {
            "u": 0.470,
            "g": 0.213,
            "r": 0.126,
            "i": 0.096,
            "z": 0.068,
            "y": 0.171,
        }
    )

    tvisRef: float | None = 30


class LsstErrorModelV2(ErrorModel):
    """Photometric error model for LSST, version 2.

    Below is the parameter docstring:
    """

    __doc__ += LsstErrorParamsV2.__doc__

    def __init__(self, **kwargs: Any) -> None:
        """Create an LSST error model.

        Keyword arguments override default values in LsstErrorParams.
        """
        super().__init__(LsstErrorParamsV2(**kwargs))
