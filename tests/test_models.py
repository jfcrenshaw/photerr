"""Tests for the error models.

The base ErrorModel object is mainly tested implicitly via LsstErrorModel.
"""

import numpy as np
import pandas as pd
import pytest

from photerr import (
    ErrorModel,
    EuclidDeepErrorModel,
    EuclidWideErrorModel,
    LsstErrorModel,
    LsstErrorModelV1,
    LsstErrorParams,
    RomanDeepErrorModel,
    RomanMediumErrorModel,
    RomanUltraDeepErrorModel,
    RomanWideErrorModel,
)


@pytest.fixture
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


@pytest.fixture
def lsst_data() -> pd.DataFrame:
    """Return 1000 random LSST galaxies for error model tests."""
    rng = np.random.default_rng(0)
    array = rng.normal(
        loc=[25, 25, 25, 25, 25, 25, 0, 0],
        scale=[2, 2, 2, 2, 2, 2, 0.2, 0.2],
        size=(10_000, 8),
    )
    array[:, -2:] = np.abs(array[:, -2:])
    dataframe = pd.DataFrame(array, columns=[*list("ugrizy"), "major", "minor"])
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
    ("sigLim", "ndMode", "absFlux"),
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
    magLimScale1 = LsstErrorModel(scale=dict.fromkeys("ugrizy", 1)).getLimitingMags()

    magLimScale2 = LsstErrorModel(scale=dict.fromkeys("ugrizy", 2)).getLimitingMags()

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
    with pytest.raises(ValueError, match="at most one positional"):
        ErrorModel({}, {})  # type: ignore
    with pytest.raises(TypeError, match="must be an ErrorParams"):
        ErrorModel({})  # type: ignore
    with pytest.raises(ValueError, match="Please provide either"):
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


def test_renameBands() -> None:
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
    assert np.allclose(ratio[1, 1:], 1)

    # but u band of low SNR is doubled
    assert np.allclose(ratio[1, 0], 2)


def test_outputType_pogson_default(lsst_data: pd.DataFrame) -> None:
    """Test that default outputType="pogson" gives the same result as before."""
    m = LsstErrorModel()
    m_explicit = LsstErrorModel(outputType="pogson")
    out1 = m(lsst_data, random_state=0)
    out2 = m_explicit(lsst_data, random_state=0)
    assert np.allclose(out1.to_numpy(), out2.to_numpy(), equal_nan=True)


def test_outputType_maggy(lsst_data: pd.DataFrame) -> None:
    """Test that outputType='maggy' is consistent with Pogson output."""
    bands = list("ugrizy")
    pogson_out = LsstErrorModel()(lsst_data, random_state=0)
    maggy_out = LsstErrorModel(outputType="maggy")(lsst_data, random_state=0)

    # keep only rows that are finite in both outputs
    finite = np.isfinite(pogson_out[bands]).all(axis=1)
    pogson_finite = pogson_out[bands][finite]
    maggy_finite = maggy_out[bands][finite]

    # converting maggies back to Pogson should match
    assert np.allclose(
        -2.5 * np.log10(maggy_finite.to_numpy()),
        pogson_finite.to_numpy(),
        rtol=1e-5,
    )

    # errors: nsr = sigma_f / f, should match pogson nsr = 10^(err/2.5) - 1
    pogson_errs = pogson_out[[f"{b}_err" for b in bands]][finite].to_numpy()
    maggy_errs = maggy_out[[f"{b}_err" for b in bands]][finite].to_numpy()
    pogson_nsr = 10 ** (pogson_errs / 2.5) - 1
    maggy_nsr = maggy_errs / maggy_finite.to_numpy()
    assert np.allclose(pogson_nsr, maggy_nsr, rtol=1e-5)


def test_outputType_maggy_highsnr(lsst_data: pd.DataFrame) -> None:
    """Test outputType='maggy' with highSNR=True hits the flux fallback path.

    With highSNR=True, obs_fluxes is None so _from_pogson derives flux from
    Pogson mags (lines 191-192 of model.py) instead of using signed observed
    fluxes.
    Consistency check: -2.5*log10(maggy) should recover the Pogson output.
    """
    bands = list("ugrizy")
    pogson_out = LsstErrorModel(highSNR=True)(lsst_data, random_state=0)
    maggy_out = LsstErrorModel(highSNR=True, outputType="maggy")(
        lsst_data, random_state=0
    )

    finite = np.isfinite(pogson_out[bands]).all(axis=1)
    assert np.allclose(
        -2.5 * np.log10(maggy_out[bands][finite].to_numpy()),
        pogson_out[bands][finite].to_numpy(),
        rtol=1e-5,
    )

    pogson_errs = pogson_out[[f"{b}_err" for b in bands]][finite].to_numpy()
    maggy_vals = maggy_out[bands][finite].to_numpy()
    maggy_errs = maggy_out[[f"{b}_err" for b in bands]][finite].to_numpy()
    assert np.allclose(maggy_errs / maggy_vals, pogson_errs, rtol=1e-5)


def test_outputType_asinh(lsst_data: pd.DataFrame) -> None:
    """Test that outputType='asinh' is consistent with Pogson output."""
    bands = list("ugrizy")
    a = 2.5 / np.log(10)

    m = LsstErrorModel(outputType="asinh")
    pogson_out = LsstErrorModel()(lsst_data, random_state=0)
    asinh_out = m(lsst_data, random_state=0)

    # b is the 1-sigma flux per band
    b = np.array([m._b[band] for band in bands])

    # keep only finite rows
    finite = np.isfinite(pogson_out[bands]).all(axis=1)
    pogson_finite = pogson_out[bands][finite].to_numpy()
    asinh_finite = asinh_out[bands][finite].to_numpy()

    # convert luptitudes back to Pogson
    flux = 2 * b * np.sinh(-asinh_finite / a - np.log(b))
    pogson_from_asinh = -2.5 * np.log10(flux)
    assert np.allclose(pogson_from_asinh, pogson_finite, rtol=1e-5)


def test_inputType_maggy(lsst_data: pd.DataFrame) -> None:
    """Test that maggy -> pogson matches pogson -> pogson."""
    # convert catalog to maggies
    bands = list("ugrizy")
    maggy_catalog = lsst_data.copy()
    maggy_catalog[bands] = 10 ** (-lsst_data[bands].to_numpy() / 2.5)

    # estimate errors in native pogson
    pogson_in = LsstErrorModel()(lsst_data, random_state=0)

    # estimate errors for maggies but return as pogson
    maggy_in = LsstErrorModel(inputType="maggy", outputType="pogson")(
        maggy_catalog, random_state=0
    )

    # catalogs should match
    assert np.allclose(
        pogson_in.to_numpy(),
        maggy_in.to_numpy(),
        equal_nan=True,
    )


def test_inputType_asinh(lsst_data: pd.DataFrame) -> None:
    """Test that asinh -> pogson matches pogson -> pogson."""
    bands = list("ugrizy")
    a = 2.5 / np.log(10)

    model = LsstErrorModel(inputType="asinh", outputType="pogson")
    b = np.array([model._b[band] for band in bands])

    # convert catalog to luptitudes
    flux = 10 ** (-lsst_data[bands].to_numpy() / 2.5)
    asinh_catalog = lsst_data.copy()
    asinh_catalog[bands] = -a * (np.arcsinh(flux / (2 * b)) + np.log(b))

    pogson_in = LsstErrorModel()(lsst_data, random_state=0)
    asinh_in = model(asinh_catalog, random_state=0)

    assert np.allclose(
        pogson_in.to_numpy(),
        asinh_in.to_numpy(),
        equal_nan=True,
    )


def test_negative_flux_preserved_for_maggy_asinh() -> None:
    """Test that negative observed fluxes are not flagged for maggy/asinh output.

    For very faint sources, noise can push the observed flux negative.
    In Pogson output this produces a non-finite mag and gets flagged;
    in maggy/asinh it should yield a valid (finite, possibly negative) value.
    """
    rng = np.random.default_rng(42)

    # Build a catalog of very faint sources where negative flux is common
    # (true flux well below the noise floor → SNR ~ 0)
    n = 2000
    catalog = pd.DataFrame(
        {"g": np.full(n, 35.0)}  # mag=35 is way below the LSST noise floor
    )

    # With Pogson output, most sources should be non-detections (np.inf)
    pogson_out = LsstErrorModel(sigLim=0)(catalog, random_state=rng)
    assert np.any(~np.isfinite(pogson_out["g"])), "expected some non-finite Pogson mags"

    # With maggy output, all values should be finite (negative flux is valid)
    maggy_out = LsstErrorModel(outputType="maggy", sigLim=0)(catalog, random_state=rng)
    assert np.all(np.isfinite(maggy_out["g"])), "maggy output should be finite"
    assert np.all(np.isfinite(maggy_out["g_err"])), "maggy errors should be finite"
    assert np.any(maggy_out["g"] < 0), "some negative fluxes expected at mag=35"

    # With asinh output, all values should also be finite
    asinh_out = LsstErrorModel(outputType="asinh", sigLim=0)(catalog, random_state=rng)
    assert np.all(np.isfinite(asinh_out["g"])), "asinh output should be finite"
    assert np.all(np.isfinite(asinh_out["g_err"])), "asinh errors should be finite"

    # Verify luptitude formula: converting back to flux should give the same
    # signed flux as maggy output (up to different random draws, so use same seed)
    rng2 = np.random.default_rng(7)
    catalog2 = pd.DataFrame({"g": np.full(500, 35.0)})
    maggy_check = LsstErrorModel(outputType="maggy", sigLim=0)(
        catalog2, random_state=rng2
    )
    rng3 = np.random.default_rng(7)
    asinh_check = LsstErrorModel(outputType="asinh", sigLim=0)(
        catalog2, random_state=rng3
    )
    m = LsstErrorModel(outputType="asinh")
    b_g = m._b["g"]
    a = 2.5 / np.log(10)
    flux_from_lup = 2 * b_g * np.sinh(-asinh_check["g"].to_numpy() / a - np.log(b_g))
    assert np.allclose(flux_from_lup, maggy_check["g"].to_numpy(), rtol=1e-5)


def test_asinhB_override() -> None:
    """Test that asinhB can be overridden per-band or with a scalar."""
    # scalar override: all bands get the same b
    m_scalar = LsstErrorModel(outputType="asinh", asinhB=1e-10)
    for band in "ugrizy":
        assert m_scalar._b[band] == 1e-10

    # dict override: only specified bands overridden
    m_dict = LsstErrorModel(outputType="asinh", asinhB={"g": 5e-11, "r": 2e-11})
    assert m_dict._b["g"] == 5e-11
    assert m_dict._b["r"] == 2e-11
    # other bands get auto-computed defaults
    assert m_dict._b["u"] != 5e-11


def test_ndFlag_with_outputTypes(data: pd.DataFrame) -> None:
    """Test that non-detections are still flagged correctly for all output types."""
    for outputType in ["pogson", "maggy", "asinh"]:
        out = LsstErrorModel(outputType=outputType, sigLim=10)(data[["g"]], 0)
        # the super-low-SNR galaxy (mag=99) should be flagged in all output types
        assert out["g"].iloc[-1] == np.inf
        assert out["g_err"].iloc[-1] == np.inf


def test_limitingMags_bad_nSigma() -> None:
    """Test that nSigma <= 0 raises a ValueError."""
    errModel = LsstErrorModel()
    with pytest.raises(ValueError, match="nSigma must be positive"):
        errModel.getLimitingMags(nSigma=0)
    with pytest.raises(ValueError, match="nSigma must be positive"):
        errModel.getLimitingMags(nSigma=-1)


def test_extended_missing_columns(data: pd.DataFrame) -> None:
    """Test that missing majorCol/minorCol raises a clear ValueError."""
    bad_catalog = data[["g", "VIS"]]  # no major/minor columns
    with pytest.raises(ValueError, match="not found in catalog"):
        LsstErrorModel(extendedType="auto")(bad_catalog, 0)
    with pytest.raises(ValueError, match="not found in catalog"):
        LsstErrorModel(extendedType="gaap")(bad_catalog, 0)


def test_limiting_mags() -> None:
    """Compare V1 limiting mags to the values in Table 2 of Ivezic 2019."""
    # get the limiting mags from the error model
    errM = LsstErrorModelV1(airmass=1)
    m5 = errM.getLimitingMags(coadded=False)

    # compare to the Ivezic 2019 values
    ivezic2019 = {
        "u": 23.78,
        "g": 24.81,
        "r": 24.35,
        "i": 23.92,
        "z": 23.34,
        "y": 22.45,
    }
    for band in m5:
        assert np.isclose(m5[band], ivezic2019[band], rtol=1e-3)


_PER_ROW_BANDS = list("ugrizy")
_PER_ROW_M5 = {band: 26.0 + 0.1 * i for i, band in enumerate(_PER_ROW_BANDS)}
_PER_ROW_PARAMS = {
    "nYrObs": 1,
    "nVisYr": 1,
    "gamma": 0.04,
    "m5": _PER_ROW_M5,
    "theta": dict.fromkeys(_PER_ROW_BANDS, 0.7),
    "airmass": dict.fromkeys(_PER_ROW_BANDS, 1.2),
}


_M5_TEMPLATES = ["m5_{band}", "{band}_depth", "depth.{band}.m5"]


@pytest.fixture
def per_row_data() -> pd.DataFrame:
    """Return galaxies plus a depth column per band, for every test template."""
    rng = np.random.default_rng(0)
    n = 500
    frame = pd.DataFrame({band: rng.uniform(20, 27, n) for band in _PER_ROW_BANDS})
    frame["major"] = rng.uniform(0.1, 1.0, n)
    frame["minor"] = frame["major"] * rng.uniform(0.5, 1.0, n)
    for band in _PER_ROW_BANDS:
        depths = _PER_ROW_M5[band] + rng.normal(0, 0.25, n)
        for template in _M5_TEMPLATES:
            frame[template.format(band=band)] = depths
    return frame


@pytest.mark.parametrize("extendedType", ["point", "auto", "gaap"])
@pytest.mark.parametrize("highSNR", [True, False])
@pytest.mark.parametrize("ndMode", ["flag", "sigLim"])
@pytest.mark.parametrize("template", _M5_TEMPLATES)
def test_m5Template_matches_magnitude_offset(
    extendedType: str,
    highSNR: bool,
    ndMode: str,
    template: str,
    per_row_data: pd.DataFrame,
) -> None:
    """Compare per-object depths against shifting the input magnitudes instead.

    Because m5 enters the error model only through (mags - m5), giving an object
    a depth of m5_row is exactly equivalent to leaving the depth at m5_ref and
    brightening the input magnitude by (m5_ref - m5_row), then undoing that
    shift on the output. That makes the shifted run an independent oracle.
    """
    extra = {
        "extendedType": extendedType,
        "highSNR": highSNR,
        "ndMode": ndMode,
        "sigLim": 1 if ndMode == "sigLim" else 0,
    }

    perRow = ErrorModel(**_PER_ROW_PARAMS, m5Template=template, **extra)(
        per_row_data, 42
    )

    # shift the input mags by (m5_ref - m5_row) and run at a single depth
    shifted = per_row_data.copy()
    offsets = {
        band: _PER_ROW_M5[band] - per_row_data[template.format(band=band)].to_numpy()
        for band in _PER_ROW_BANDS
    }
    for band in _PER_ROW_BANDS:
        shifted[band] = per_row_data[band].to_numpy() + offsets[band]
    oracle = ErrorModel(**_PER_ROW_PARAMS, **extra)(shifted, 42)

    for band in _PER_ROW_BANDS:
        expected = oracle[band].to_numpy() - offsets[band]
        actual = perRow[band].to_numpy()

        # non-detections must land on the same objects
        assert np.array_equal(np.isfinite(actual), np.isfinite(expected))

        finite = np.isfinite(actual)
        assert np.allclose(actual[finite], expected[finite])

        # errors are invariant under the shift, so they match without undoing it
        actualErr = perRow[f"{band}_err"].to_numpy()
        expectedErr = oracle[f"{band}_err"].to_numpy()
        finiteErr = np.isfinite(actualErr)
        assert np.allclose(actualErr[finiteErr], expectedErr[finiteErr])


def test_m5Template_deepens_and_shallows(per_row_data: pd.DataFrame) -> None:
    """Test that per-object depths actually change the errors, in the right way."""
    perRow = ErrorModel(**_PER_ROW_PARAMS, m5Template="m5_{band}")(per_row_data, 1)
    flat = ErrorModel(**_PER_ROW_PARAMS)(per_row_data, 1)

    # the two must differ, or m5Template is being silently ignored
    assert not np.allclose(perRow["r_err"], flat["r_err"])

    # at fixed magnitude, a deeper object must get a smaller error
    n = 200
    fixed = pd.DataFrame({band: np.full(n, 24.0) for band in _PER_ROW_BANDS})
    for band in _PER_ROW_BANDS:
        fixed[f"m5_{band}"] = _PER_ROW_M5[band]
    fixed["m5_r"] = np.linspace(24.5, 27.0, n)
    scan = ErrorModel(**_PER_ROW_PARAMS, m5Template="m5_{band}", decorrelate=False)(
        fixed, 3
    )
    errs = scan["r_err"].to_numpy()
    assert np.all(np.diff(errs) < 0)

    # and a band without varying depth must be untouched by that scan
    assert np.allclose(scan["u_err"], scan["u_err"].iloc[0])


def test_m5Template_missing_columns(per_row_data: pd.DataFrame) -> None:
    """Test that a prefix matching no column raises rather than falling back."""
    with pytest.raises(ValueError, match="none of the expected depth"):
        ErrorModel(**_PER_ROW_PARAMS, m5Template="nope_{band}")(per_row_data, 0)


def test_m5Template_partial_coverage(per_row_data: pd.DataFrame) -> None:
    """Test that bands without a depth column fall back to the model's depth."""
    partial = per_row_data.drop(columns=[f"m5_{band}" for band in "ugizy"])
    perRow = ErrorModel(**_PER_ROW_PARAMS, m5Template="m5_{band}")(partial, 7)
    flat = ErrorModel(**_PER_ROW_PARAMS)(per_row_data, 7)

    # r has a depth column, so it moves
    assert not np.allclose(perRow["r_err"], flat["r_err"])

    # the others do not, so they match the flat model exactly
    for band in "ugizy":
        assert np.allclose(perRow[f"{band}_err"], flat[f"{band}_err"], equal_nan=True)


def test_m5Template_leaves_model_properties_alone(per_row_data: pd.DataFrame) -> None:
    """Test that m5Template does not disturb getLimitingMags or the asinh softening.

    Both describe the model rather than a catalog, so they must stay scalar and
    keep using the model's own depths.
    """
    params = {**_PER_ROW_PARAMS, "sigmaSys": 0}
    withPrefix = ErrorModel(**params, m5Template="m5_{band}", outputType="asinh")
    without = ErrorModel(**params, outputType="asinh")

    assert withPrefix.getLimitingMags() == without.getLimitingMags()
    assert withPrefix._b == without._b

    # single-visit limiting mags still round-trip to the m5 that was passed in
    single = withPrefix.getLimitingMags(coadded=False)
    for band, m5 in _PER_ROW_M5.items():
        assert np.ndim(single[band]) == 0
        assert np.isclose(single[band], m5)

    # asinh output still runs, and still responds to the per-object depths
    perRow = withPrefix(per_row_data, 1)
    flat = without(per_row_data, 1)
    assert not np.allclose(perRow["r"], flat["r"])


def test_m5Template_orientation_when_nrows_equals_nbands() -> None:
    """Test that a square catalog is not transposed into the band axis.

    Stacking the depths along the leading axis instead of the trailing one is
    undetectable except when the number of rows equals the number of bands, so
    check that case directly.
    """
    n = len(_PER_ROW_BANDS)
    square = pd.DataFrame({band: np.full(n, 24.0) for band in _PER_ROW_BANDS})
    for band in _PER_ROW_BANDS:
        square[f"m5_{band}"] = np.linspace(25.0, 27.0, n)
    out = ErrorModel(**_PER_ROW_PARAMS, m5Template="m5_{band}", decorrelate=False)(
        square, 1
    )

    # every band's depth increases down the rows, so every error must decrease
    for band in _PER_ROW_BANDS:
        assert np.all(np.diff(out[f"{band}_err"].to_numpy()) < 0)


@pytest.mark.parametrize("template", ["{band}", "lsst_{band}"])
def test_m5Template_rejects_band_collision(template: str) -> None:
    """Test that a template resolving onto a band column is rejected.

    A bare "{band}" template would read the magnitude column as the depth,
    which is silently wrong rather than obviously wrong.
    """
    bands = ["u", "lsst_u"]
    params = {
        "nYrObs": 1,
        "nVisYr": 1,
        "gamma": 0.04,
        "m5": dict.fromkeys(bands, 26.0),
    }
    with pytest.raises(ValueError, match="are themselves band columns"):
        ErrorModel(**params, m5Template=template)


@pytest.mark.parametrize(
    "template", ["m5_all", "m5_{filter}", "m5_{band}_{extra}", "m5_{}"]
)
def test_m5Template_requires_band_field(template: str) -> None:
    """Test that a template without exactly one {band} field is rejected.

    Without it, every band would resolve to the same column, so every band
    would silently share one depth.
    """
    with pytest.raises(ValueError, match=r"must contain '\{band\}'"):
        ErrorModel(nYrObs=1, nVisYr=1, gamma=0.04, m5={"u": 26.0}, m5Template=template)


def test_m5Template_rejects_malformed_format_string() -> None:
    """Test that unbalanced braces give a format-string error, not a crash."""
    with pytest.raises(ValueError, match="not a valid format string"):
        ErrorModel(nYrObs=1, nVisYr=1, gamma=0.04, m5={"u": 26.0}, m5Template="m5_{ba")
