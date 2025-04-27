"""Tests for the error models.

The base ErrorModel object is mainly tested implicitly via LsstErrorModel.
"""

import numpy as np
import pandas as pd
import pytest

from photerr import (
    ErrorModel,
    EuclidWideErrorModel,
    EuclidDeepErrorModel,
    LsstErrorModel,
    LsstErrorModelV1,
    LsstErrorParams,
    RomanWideErrorModel,
    RomanMediumErrorModel,
    RomanDeepErrorModel,
    RomanUltraDeepErrorModel,
)


@pytest.fixture()
def data() -> pd.DataFrame:
    """Return dummy data for error model tests.

    Includes a high SNR, a low SNR, and a super low SNR galaxy.
    Includes an LSST band, a Euclid band, and a Euclid+Roman band, Roman deep band.
    """
    array = np.array(
        [
            [21, 21, 21, 21, 0.02, 0.01],  # high SNR
            [28, 28, 28, 28, 0.2, 0.1],  # low SNR
            [99, 99, 99, 99, 2, 1],  # super low SNR
        ]
    )
    dataframe = pd.DataFrame(array, columns=["g", "VIS", "J", "W", "major", "minor"])
    return dataframe


@pytest.fixture()
def lsst_data() -> pd.DataFrame:
    """Return 1000 random LSST galaxies for error model tests."""
    rng = np.random.default_rng(0)
    array = rng.normal(
        loc=[25, 25, 25, 25, 25, 25, 0, 0],
        scale=[2, 2, 2, 2, 2, 2, 0.2, 0.2],
        size=(10_000, 8),
    )
    array[:, -2:] = np.abs(array[:, -2:])
    dataframe = pd.DataFrame(array, columns=list("ugrizy") + ["major", "minor"])
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
        LsstErrorModel()(data, "fake")  # type: ignore


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
    default_errs = LsstErrorModelV1()(data, random_state=0)
    greater_errs = LsstErrorModelV1(**params)(data, random_state=0)
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


def test_pointSource_sigLim(data: pd.DataFrame) -> None:
    """Test that everything beyond sigLim is flagged."""
    assert np.all(LsstErrorModel(sigLim=10)(data, 0)[["g", "g_err"]].iloc[1:] == np.inf)


@pytest.mark.parametrize("extendedType", ["auto", "gaap"])
@pytest.mark.parametrize("ndMode", ["flag", "sigLim"])
@pytest.mark.parametrize("highSNR", [True, False])
@pytest.mark.parametrize("decorrelate", [True, False])
def test_extended_sigLim(
    extendedType: str,
    ndMode: str,
    highSNR: bool,
    decorrelate: bool,
    lsst_data: pd.DataFrame,
) -> None:
    """Test that sigLim works with extended source errors."""
    # generate the photometric errors
    sigLim = 3.0
    errM = LsstErrorModel(
        sigLim=sigLim,
        ndMode=ndMode,
        extendedType=extendedType,
        highSNR=highSNR,
        decorrelate=decorrelate,
    )
    data = errM(lsst_data, 0)

    # keep only the galaxies with finite magnitudes
    data = data[np.isfinite(data).all(axis=1)]

    # calculate the SNR of every galaxy
    if highSNR:
        snr = 1 / data[[col for col in data.columns if "err" in col]]
    else:
        snr = 1 / (
            10 ** (data[[col for col in data.columns if "err" in col]] / 2.5) - 1
        )
    snr.rename(columns=lambda name: name.replace("err", "snr"), inplace=True)

    # make sure the minimum SNR exceeds sigLim
    assert np.all((snr.min() >= sigLim) | np.isclose(snr.min(), sigLim))


@pytest.mark.parametrize("highSNR", [True, False])
@pytest.mark.parametrize("decorrelate", [True, False])
def test_pointSource_ndFlag_sigLim(
    highSNR: bool, decorrelate: bool, data: pd.DataFrame
) -> None:
    """Test that ndFlag=sigLim works for point sources.

    i.e. everything beyond sigLim is cut to sigLim when ndMode==sigLim
    """
    sigLimData = LsstErrorModel(
        sigLim=10,
        ndMode="sigLim",
        highSNR=highSNR,
        decorrelate=decorrelate,
    )(data, 0)
    assert np.isclose(sigLimData["g"][1], sigLimData["g"][2])
    assert np.isclose(sigLimData["g_err"][1], sigLimData["g_err"][2])


@pytest.mark.parametrize("highSNR", [False, True])
def test_decorrelate_doesnt_change_siglim(highSNR: bool, data: pd.DataFrame) -> None:
    """Test that decorrelate doesn't change the sigLim-ed outputs."""
    decorrData = LsstErrorModel(
        sigLim=100, ndMode="sigLim", highSNR=highSNR, decorrelate=True
    )(data, 0)

    corrData = LsstErrorModel(
        sigLim=100, ndMode="sigLim", highSNR=highSNR, decorrelate=False
    )(data, 0)

    assert np.allclose(decorrData[1:], corrData[1:])


@pytest.mark.parametrize("decorrelate", [False, True])
@pytest.mark.parametrize(
    "sigLim,ndMode,absFlux",
    [
        (5, "sigLim", False),
        (0, "flag", True),
    ],
)
def test_finite_values(
    decorrelate: bool,
    sigLim: float,
    ndMode: str,
    absFlux: bool,
    lsst_data: pd.DataFrame,
) -> None:
    """Test settings that should result in all finite values.

    These two settings should result in all finite values:
        - absFlux=True with sigLim=0
        - ndMode="sigLim" with sigLim>0
    """
    # first let's make sure that our default has some non-finite elements
    assert ~np.all(
        np.isfinite(
            LsstErrorModel(
                decorrelate=decorrelate,
                sigLim=sigLim,
            )(lsst_data, 0)
        )
    )

    # now check that enabling these settings makes everything finite
    assert np.all(
        np.isfinite(
            LsstErrorModel(
                decorrelate=decorrelate,
                sigLim=sigLim,
                ndMode=ndMode,
                absFlux=absFlux,
            )(lsst_data, 0)
        )
    )


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


@pytest.mark.parametrize("extendedType", ["auto", "gaap"])
def test_limitMags_aperture(extendedType: str) -> None:
    """Test that increasing the aperture decreasing depth."""
    errModel = LsstErrorModel(extendedType=extendedType)

    magLimAp0 = errModel.getLimitingMags(aperture=0)
    magLimAp1 = errModel.getLimitingMags(aperture=1)
    assert all(magLimAp0[band] > magLimAp1[band] for band in magLimAp0)


def test_limitingMags_scale() -> None:
    """Test that increasing the error scale decreases the limiting mags."""
    magLimScale1 = LsstErrorModel(
        scale={band: 1 for band in "ugrizy"}
    ).getLimitingMags()

    magLimScale2 = LsstErrorModel(
        scale={band: 2 for band in "ugrizy"}
    ).getLimitingMags()

    assert all(magLimScale1[band] > magLimScale2[band] for band in magLimScale1)


def test_errLoc(data: pd.DataFrame) -> None:
    """Test that errLoc works as expected."""
    # the error column should come right after the magnitude column
    after = LsstErrorModel(errLoc="after")(data, 0)
    assert list(after.columns) == ["g", "g_err", "VIS", "J", "W", "major", "minor"]

    # the error column should come at the end
    end = LsstErrorModel(errLoc="end")(data, 0)
    assert list(end.columns) == ["g", "VIS", "J", "W", "major", "minor", "g_err"]

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


def test_other_models(data: pd.DataFrame) -> None:
    """Test instantiating other models and calculating errors."""

    lsstData = LsstErrorModelV1()(data, 0)
    assert lsstData.shape == (data.shape[0], data.shape[1] + 1)

    euclidData = EuclidWideErrorModel()(data, 0)
    assert euclidData.shape == (data.shape[0], data.shape[1] + 2)

    euclidData = EuclidDeepErrorModel()(data, 0)
    assert euclidData.shape == (data.shape[0], data.shape[1] + 2)

    romanData = RomanWideErrorModel()(data, 0)
    assert romanData.shape == (data.shape[0], data.shape[1] + 0)

    romanData = RomanMediumErrorModel()(data, 0)
    assert romanData.shape == (data.shape[0], data.shape[1] + 1)

    romanData = RomanDeepErrorModel()(data, 0)
    assert romanData.shape == (data.shape[0], data.shape[1] + 2)

    romanData = RomanUltraDeepErrorModel()(data, 0)
    assert romanData.shape == (data.shape[0], data.shape[1] + 1)


def test_rename_bands() -> None:
    """Test renaming the bands in the error model.

    This failure was noticed by Sam Schmidt.
    """
    LsstErrorModel(renameDict={"y": "y_lsst"})
    EuclidWideErrorModel(renameDict={"Y": "y_euclid"})
    RomanMediumErrorModel(renameDict={"Y": "y_roman"})


def test_alternate_instantiation() -> None:
    """Test alternate ways of instantiating models."""
    # test that all arguments works and gives me same effect as Params object
    errM1 = ErrorModel(LsstErrorParams())
    errM2 = ErrorModel(**LsstErrorParams().__dict__)
    assert errM1.params.__dict__ == errM2.params.__dict__

    # test that I can update params inside or outside the param object
    errM1 = ErrorModel(LsstErrorParams(), nYrObs=2)
    errM2 = ErrorModel(LsstErrorParams(nYrObs=2))
    assert errM1.params == errM2.params


@pytest.mark.parametrize("extendedType", ["point", "auto", "gaap"])
@pytest.mark.parametrize("highSNR", [True, False])
def test_mag_nsr_inversion(
    extendedType: str,
    highSNR: bool,
    lsst_data: pd.DataFrame,
) -> None:
    """Test that the mag->nsr and nsr->mag methods are inverses."""
    # get the arrays of data
    bands = list("ugrizy")
    mags = lsst_data[bands].to_numpy()
    majors = lsst_data["major"].to_numpy()
    minors = lsst_data["minor"].to_numpy()

    # create the error model
    errM = LsstErrorModel(extendedType=extendedType, highSNR=highSNR)

    # compute NSR
    nsr = errM._get_nsr_from_mags(mags, majors, minors, bands)

    # use these NSRs to re-compute input mags
    mags_recomputed = errM._get_mags_from_nsr(nsr, majors, minors, bands)

    # make sure we got back what we put in
    assert np.allclose(mags, mags_recomputed)


@pytest.mark.parametrize("sigLim", [0, 5])
def test_both_not_finite(sigLim: float, lsst_data: pd.DataFrame) -> None:
    """Test that the band and error are both finite at the same time."""
    data = LsstErrorModel(sigLim=sigLim)(lsst_data, 0)

    # loop through every band
    for band in "ugrizy":
        # where band is not finite, make sure error is not finite
        assert ~np.isfinite(data[f"{band}_err"][~np.isfinite(data[band])]).any()

        # where error is not finite, make sure band is not finite
        assert ~np.isfinite(data[band][~np.isfinite(data[f"{band}_err"])]).any()


@pytest.mark.parametrize("highSNR", [True, False])
def test_scale(highSNR: bool) -> None:
    """Test that scale linearly scales the final errors and changes limiting mags.

    Note we have to set decorrelate=False to make sure the errors are re-sampled,
    and absFlux=True to make sure all the errors are finite.
    """
    # some custom data for this test
    # one super high SNR, and one very low SNR
    data = pd.DataFrame(
        np.array([[10], [30]]) * np.ones((2, 6)),
        columns=list("ugrizy"),
    )

    # calculate errors with u scale = 1 and 2
    errsScale1 = LsstErrorModel(
        errLoc="alone",
        decorrelate=False,
        absFlux=True,
        highSNR=highSNR,
    )(data, 0)
    errsScale2 = LsstErrorModel(
        errLoc="alone",
        decorrelate=False,
        absFlux=True,
        highSNR=highSNR,
        scale={"u": 2},
    )(data, 0)

    # convert to numpy array
    errsScale1 = errsScale1.to_numpy()
    errsScale2 = errsScale2.to_numpy()

    # calculate nsr
    if highSNR:
        nsrScale1 = errsScale1
        nsrScale2 = errsScale2
    else:
        nsrScale1 = 10 ** (errsScale1 / 2.5) - 1
        nsrScale2 = 10 ** (errsScale2 / 2.5) - 1

    # calculate the error ratio
    ratio = nsrScale2 / nsrScale1

    # check that super high SNR is not impacted
    assert np.allclose(ratio[0, :], 1)

    # and the grizy bands of the low SNR
    np.allclose(ratio[1, 1:], 1)

    # but u band of low SNR is doubled
    np.allclose(ratio[1, 0], 2)


def test_limiting_mags() -> None:
    """Compare V1 limiting mags to the values in Table 2 of Ivezic 2019."""
    # get the limiting mags from the error model
    errM = LsstErrorModelV1(airmass=1)
    m5 = errM.getLimitingMags(coadded=False)

    # compare to the Ivezic 2019 values
    ivezic2019 = dict(u=23.78, g=24.81, r=24.35, i=23.92, z=23.34, y=22.45)
    for band in m5:
        assert np.isclose(m5[band], ivezic2019[band], rtol=1e-3)
