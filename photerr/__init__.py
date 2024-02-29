# flake8: noqa I252, F401
from importlib import metadata

from .euclid import EuclidErrorModel, EuclidErrorParams
from .lsstV1 import LsstErrorModelV1, LsstErrorParamsV1
from .lsstV2 import LsstErrorModelV2, LsstErrorParamsV2
from .model import ErrorModel
from .params import ErrorParams
from .roman import RomanErrorModel, RomanErrorParams

# set the version number
__version__ = metadata.version(__package__)
del metadata


# alias the latest LSST error model
class LsstErrorParams(LsstErrorParamsV2): ...


class LsstErrorModel(LsstErrorModelV2): ...
