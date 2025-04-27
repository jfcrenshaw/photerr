# flake8: noqa I252, F401
from importlib import metadata

from .euclid import EuclidWideErrorModel as EuclidErrorModel
from .euclid import EuclidWideErrorParams as EuclidErrorParams
from .euclid import *
from .lsstV1 import *
from .lsstV2 import LsstErrorModelV2 as LsstErrorModel
from .lsstV2 import LsstErrorParamsV2 as LsstErrorParams
from .lsstV2 import *
from .model import ErrorModel
from .params import ErrorParams
from .roman import RomanMediumErrorModel as RomanErrorModel
from .roman import RomanMediumErrorParams as RomanErrorParams
from .roman import *

# set the version number
__version__ = metadata.version(__package__)
del metadata
