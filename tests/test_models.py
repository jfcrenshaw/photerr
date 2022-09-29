"""Tests for the error models.

The base ErrorModel object is mainly tested implicitly via LsstErrorModel.
"""
import numpy as np
import pandas as pd
import pytest

from photerr import ErrorModel, EuclidErrorModel, LsstErrorModel, RomanErrorModel


@pytest.fixture()
def data() -> pd.DataFrame:
    """Return dummy data for error model tests.

    Includes a high SNR, a low SNR, and a super low SNR galaxy.
    Includes an LSST band, a Euclid+Roman band, and a Roman band.
    """
    array = np.array(
        [
            [21, 21, 21, 0.02, 0.01],  # high SNR
            [28, 28, 28, 0.2, 0.1],  # low SNR
            [99, 99, 99, 2, 1],  # super low SNR
        ]
    )
    dataframe = pd.DataFrame(array, columns=["g", "J", "F", "major", "minor"])
    return dataframe


def test_random_state(data: pd.DataFrame) -> None:
    """Test that the random state behaves as expected."""
    data = data[["g"]].iloc[:2]
    errModel = LsstErrorModel()

    # same seed results in same results
    assert np.allclose(errModel(data, 0), errModel(data, 0))
    assert np.allclose(errModel(data, 0), errModel(data, np.random.default_rng(0)))

    # different seeds result in different results
    assert ~np.isclose(errModel(data, 0), errModel(data, 1)).any()
    assert ~np.isclose(
        errModel(data, 0),
        errModel(data, np.random.default_rng(1)),
    ).any()

    # bad random state should cause error
    with pytest.raises(TypeError):
        LsstErrorModel()(data, "fake")


@pytest.mark.parametrize(
    "params",
    [
        ({"nYrObs": 0.5}),
        ({"nVisYr": {"g": 1}}),
        ({"gamma": {"g": 0.1}}),
        ({"m5": {"g": 20}}),
        ({"tvis": 1}),
        ({"airmass": 1.5}),
        ({"Cm": {"g": 20}}),
        ({"msky": {"g": 5}}),
        ({"theta": {"g": 1}}),
        ({"km": {"g": 0.5}}),
        ({"sigmaSys": 0.1}),
        ({"extendedType": "auto"}),
        ({"extendedType": "gaap"}),
        ({"extendedType": "auto", "aMin": 1.5}),
        ({"extendedType": "gaap", "aMin": 1.5}),
    ],
)
def test_increasing_error(params: dict, data: pd.DataFrame) -> None:
    """Test that error magnitudes respond to parameters as expected."""
    data = data.iloc[:2]
    default_errs = LsstErrorModel()(data, random_state=0)
    greater_errs = LsstErrorModel(**params)(data, random_state=0)
    assert all(greater_errs["g_err"] > default_errs["g_err"])


def test_decorrelate(data: pd.DataFrame) -> None:
    """Test that decorrelate makes the photometric error non-deterministic."""
    data = data.iloc[:2]

    # when decorrelate = False, the errors are always the same, regardless of the seed
    corrErrModel = LsstErrorModel(decorrelate=False)
    correlated1 = corrErrModel(data, random_state=1)
    correlated2 = corrErrModel(data, random_state=2)
    assert np.allclose(correlated1["g_err"], correlated2["g_err"])

    # when decorrelate = True, errors are different with different seeds
    decorrErrModel = LsstErrorModel(decorrelate=True)
    decorrelated1 = decorrErrModel(data, random_state=1)
    decorrelated2 = decorrErrModel(data, random_state=2)
    assert ~np.isclose(decorrelated1["g_err"], decorrelated2["g_err"]).any()


def test_SNR_regimes(data: pd.DataFrame) -> None:
    """Test that highSNR is same in high SNR regime, but greater in low SNR  regime.

    highSNR=True/False should not matter for high-SNR galaxies, but for low-SNR
    galaxies, highSNR=True should give larger errors.
    """
    # setup both models
    lowSNRmodel = LsstErrorModel(highSNR=False, decorrelate=False)
    highSNRmodel = LsstErrorModel(highSNR=True, decorrelate=False)

    # calculate errors using both models
    lowSNRerrs = lowSNRmodel(data, 0)
    highSNRerrs = highSNRmodel(data, 0)

    # check that errors for high SNR galaxy are very close
    assert np.isclose(
        highSNRerrs["g_err"].iloc[0], lowSNRerrs["g_err"].iloc[0], rtol=1e-2
    )

    # check that errors for low SNR galaxy are greater
    assert ~np.isclose(
        highSNRerrs["g_err"].iloc[1], lowSNRerrs["g_err"].iloc[1], rtol=1e-2
    )
    assert highSNRerrs["g_err"].iloc[1] > lowSNRerrs["g_err"].iloc[1]


def test_ndFlag(data: pd.DataFrame) -> None:
    """Test different values for ndFlag."""
    assert LsstErrorModel(ndFlag=np.inf, sigLim=1)(data, 0).iloc[-1, 0] == np.inf
    assert np.isnan(LsstErrorModel(ndFlag=np.nan, sigLim=1)(data, 0).iloc[-1, 0])
    assert LsstErrorModel(ndFlag=999, sigLim=1)(data, 0).iloc[-1, 0] == 999
    assert LsstErrorModel(ndFlag=-999, sigLim=1)(data, 0).iloc[-1, 0] == -999


def test_sigLim(data: pd.DataFrame) -> None:
    """Test that sigLim works correctly."""
    # test that everything beyond sigLim is flagged
    assert np.all(LsstErrorModel(sigLim=10)(data, 0)[["g", "g_err"]].iloc[1:] == np.inf)

    # test that everything beyond sigLim is cut to sigLim when ndMode==sigLim
    # first with highSNR=False
    sigLimData = LsstErrorModel(sigLim=10, ndMode="sigLim", highSNR=False)(data, 0)
    assert np.isclose(sigLimData["g"][1], sigLimData["g"][2])
    assert np.isclose(sigLimData["g_err"][1], sigLimData["g_err"][2])
    # now with highSNR=True
    sigLimData = LsstErrorModel(sigLim=10, ndMode="sigLim", highSNR=True)(data, 0)
    assert np.isclose(sigLimData["g"][1], sigLimData["g"][2])
    assert np.isclose(sigLimData["g_err"][1], sigLimData["g_err"][2])


def test_absFlux(data: pd.DataFrame) -> None:
    """Test that absFlux results in all finite magnitudes."""
    assert ~np.all(np.isfinite(LsstErrorModel(absFlux=False)(data, 5)))
    assert np.all(np.isfinite(LsstErrorModel(absFlux=True)(data, 5)))


def test_limitingMags() -> None:
    """Test that getLimitingMags behaves as expected."""
    errModel = LsstErrorModel()

    # coadded depths are deeper
    magLim = errModel.getLimitingMags(coadded=False)
    magLimCoadd = errModel.getLimitingMags(coadded=True)
    assert all(magLimCoadd[band] > magLim[band] for band in magLim)

    # 1sig is deeper than 10sig
    magLim1sig = errModel.getLimitingMags(nSigma=1)
    magLim10sig = errModel.getLimitingMags(nSigma=10)
    assert all(magLim1sig[band] > magLim10sig[band] for band in magLim1sig)

    # 5sig without coadd and without system error matches m5 values
    errModel = LsstErrorModel(
        m5={"u": 23, "g": 24, "r": 25, "i": 26, "z": 27, "y": 28},
        sigmaSys=0,
    )
    magLim = errModel.getLimitingMags(coadded=False)
    assert all(np.isclose(errModel.params.m5[band], magLim[band]) for band in magLim)


def test_errLoc(data: pd.DataFrame) -> None:
    """Test that errLoc works as expected."""
    # the error column should come right after the magnitude column
    after = LsstErrorModel(errLoc="after")(data, 0)
    assert list(after.columns) == ["g", "g_err", "J", "F", "major", "minor"]

    # the error column should come at the end
    end = LsstErrorModel(errLoc="end")(data, 0)
    assert list(end.columns) == ["g", "J", "F", "major", "minor", "g_err"]

    # the error column should be alone
    alone = LsstErrorModel(errLoc="alone")(data, 0)
    assert list(alone.columns) == ["g_err"]


def test_bad_instantiation() -> None:
    """Test some bad inputs to instantiation."""
    with pytest.raises(ValueError):
        ErrorModel({}, {})  # type: ignore
    with pytest.raises(TypeError):
        ErrorModel({})  # type: ignore
    with pytest.raises(ValueError):
        ErrorModel()


def test_other_models(data) -> None:
    """Test instantiating other models and calculating errors."""
    euclidData = EuclidErrorModel()(data, 0)
    assert euclidData.shape == (data.shape[0], data.shape[1] + 1)

    romanData = RomanErrorModel()(data, 0)
    assert romanData.shape == (data.shape[0], data.shape[1] + 2)
