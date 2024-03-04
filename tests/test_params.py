"""Tests for the error param objects.

The base ErrorParams object is only tested implicitly via LsstErrorParams.
"""

import numpy as np
import pytest

from photerr import LsstErrorParams
from photerr.params import _val_dict


def test_clean_dictionaries() -> None:
    """Test whether the parameter dictionaries are properly cleaned."""
    # first we will specify nVisYr only for g, which should force the removal of
    # the other bands
    for key, param in LsstErrorParams(nVisYr={"g": 8}).__dict__.items():
        # for each dictionary parameter
        if isinstance(param, dict):
            # m5 should be empty
            if key == "m5":
                assert len(param) == 0
            # the other parameters should only contain g
            else:
                assert list(param.keys()) == ["g"]

    # next we will check that specifying m5 removes entries from relevant dictionaries
    for key, param in LsstErrorParams(m5={"g": 28}).__dict__.items():
        if isinstance(param, dict) and key in ["Cm", "msky", "km"]:
            assert "g" not in param


def test_no_bands_left() -> None:
    """Test that dictionary params without sufficient overlap results in no bands."""
    with pytest.raises(ValueError):
        LsstErrorParams(nVisYr={"K": 10})


def test_rename_bands() -> None:
    """Test that band renaming works properly."""
    renameDict = {band: f"lsst_{band}" for band in "ugrizy"}

    # test that all of the dictionaries are renamed
    for param in LsstErrorParams(renameDict=renameDict).__dict__.values():
        if isinstance(param, dict) and len(param) > 0:
            assert set(param.keys()) == set(renameDict.values())

    # test that you can set other parameters using the old or new band names
    with_old = LsstErrorParams(renameDict=renameDict, nVisYr={"u": 1})
    with_new = LsstErrorParams(renameDict=renameDict, nVisYr={"lsst_u": 1})
    assert with_old == with_new


def test_copy() -> None:
    """Test the parameter copy method works as expected."""
    # test the two objects are identical
    params1 = LsstErrorParams()
    params2 = params1.copy()
    assert params1 == params2

    # test changing one object doesn't change the other
    params2.update(m5={"r": 30})
    assert params1 != params2


def test_update() -> None:
    """Test the update method."""
    # test passing a non-dictionary fails
    with pytest.raises(TypeError):
        LsstErrorParams().update(1)  # type: ignore

    # test passing multiple arguments fails
    with pytest.raises(ValueError):  # noqa: PT011
        LsstErrorParams().update({}, {})

    # test passing fake parameters fails
    with pytest.raises(ValueError):
        LsstErrorParams().update(fake=True)

    # test that all of these give the same results
    params1 = LsstErrorParams(nVisYr={"u": 20}, m5={"r": 30})

    params2 = LsstErrorParams()
    params2.update(nVisYr={"u": 20}, m5={"r": 30})

    params3 = LsstErrorParams()
    params3.update(dict(nVisYr={"u": 20}, m5={"r": 30}))

    assert params1 == params2
    assert params1 == params3
    assert params2 == params3

    # test that an update that results in no bands left leaves the params unchanged
    params1 = LsstErrorParams()
    params2 = params1.copy()  # type: ignore
    with pytest.raises(ValueError):  # noqa: PT011
        params2.update(nVisYr={"K": 10})
    assert params1 == params2


def test_param_val_dict() -> None:
    """Make sure that the params _val_dict is comprehensive."""
    assert set(_val_dict.keys()) == set(LsstErrorParams().__dict__.keys())


@pytest.mark.parametrize(
    "params,error",
    [
        ({"nVisYr": "test"}, TypeError),
        ({"nYrObs": {}}, TypeError),
        ({"ndFlag": "test"}, TypeError),
        ({"extendedType": "test"}, ValueError),
        ({"nYrObs": -1}, ValueError),
        ({"extendedType": "auto", "theta": {}}, ValueError),
        ({"extendedType": "auto", "aMin": 3, "aMax": 2}, ValueError),
        ({"renameDict": -1}, TypeError),
    ],
)
def test_bad_params(params: dict, error: Exception) -> None:
    """Test that instantiation and updating fails with bad parameters."""
    with pytest.raises(error):  # type: ignore
        LsstErrorParams(**params)
    with pytest.raises(error):  # type: ignore
        LsstErrorParams().update(**params)


def test_no_validation() -> None:
    """Test that without parameter validation, you can assign bogus params."""
    # with validation, this raises a type error
    with pytest.raises(TypeError):
        LsstErrorParams(tvis="fake")  # type: ignore
    # without validation, no problem!
    LsstErrorParams(tvis="fake", validate=False)  # type: ignore


def test_missing_theta() -> None:
    """Test fail if we have extended error but don't have theta for everyone."""
    with pytest.raises(ValueError):
        LsstErrorParams(extendedType="auto", theta={"g": 0.1}, m5={"u": 23})


def test_all_dicts_are_floats() -> None:
    """Test that instantiation fails if all dictionaries are floats."""
    with pytest.raises(ValueError):
        LsstErrorParams(
            nVisYr=1,
            gamma=1,
            m5=1,
            tvis=1,
            airmass=1,
            Cm=1,
            dCmInf=1,
            msky=1,
            mskyDark=1,
            theta=1,
            km=1,
        )


def test_validate_params_with_numpy_float() -> None:
    LsstErrorParams(m5={"u": np.array([23.0])[0]})
