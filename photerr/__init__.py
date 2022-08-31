# flake8: noqa I252, F401
from importlib import metadata

from .euclid import EuclidErrorModel, EuclidErrorParams
from .lsst import LsstErrorModel, LsstErrorParams
from .model import PhotometricErrorModel
from .params import ErrorParams
from .roman import RomanErrorModel, RomanErrorParams

__version__ = metadata.version(__package__)
del metadata
