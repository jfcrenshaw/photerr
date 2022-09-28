<div align="center">

[![build](https://github.com/jfcrenshaw/photerr/actions/workflows/main.yml/badge.svg)](https://github.com/jfcrenshaw/photerr/actions/workflows/main.yml)
[![Codecov](https://img.shields.io/codecov/c/github/jfcrenshaw/photerr?label=codecov&logo=codecov)](https://app.codecov.io/gh/jfcrenshaw/photerr)
[![PyPI](https://img.shields.io/pypi/v/photerr?color=blue&label=PyPI)](https://pypi.org/project/photerr/)

</div>

# PhotErr

PhotErr is a photometric error model for astronomical imaging surveys.
It implements a generalization of the high-SNR point-source error model from [Ivezic (2019)](https://arxiv.org/abs/0805.2366) that is more accurate in the low SNR regime and includes errors for extended sources, using the models from [van den Busch (2020)](http://arxiv.org/abs/2007.01846) and [Kuijken (2019)](https://arxiv.org/abs/1902.11265).

PhotErr currently includes photometric error models for the Vera C. Rubin Observatory Legacy Survey of Space and Time (LSST), as well as the Euclid and Nancy Grace Roman space telescopes.

# Getting started

PhotErr is available on PyPI and can be installed with pip:

```bash
pip install photerr
```

Note that PhotErr requires Python >= 3.8.

Once installed, you can import the error models. For example, to use the default LSST error model,

```python
from photerr import LsstErrorModel
errModel = LsstErrorModel()
catalog_with_errors = errModel(catalog, random_state=42)
```

The error model expects an input catalog in the form of a pandas DataFrame with true magnitudes, and it returns another DataFrame containing observed magnitudes and photometric errors.
Any extraneous columns in the DataFrame (e.g. a redshift column), remain in the new DataFrame - their presence does not effect the error model.

*If compatibility with Astropy Tables, Ordered Dictionaries, etc., would be useful to you, let me know!*

# Tweaking the error model

There are many parameters you can tweak to fine tune the error model.
To see all available parameters, you can either call help on the error model's `params` object,

```python
help(errModel.params)
```

or look at the docstring of the corresponding parameters object,

```python
from photerr import LsstErrorParams
help(LsstErrorParams)
```

All model parameters can be overridden using keyword arguments to the error model constructor.
Below, we explain a few of the more commonly tweaked parameters.

### *Changing the observing duration*

The example above uses the default settings for the LSST model, which includes 10 years of observing time.
If instead you want to calculate errors for LSST year 1, you can pass the `nYrObs` argument to the constructor:

```python
errModel = LsstErrorModel(nYrObs=1)
```

### *Changing the band names*

Another parameter you might want to tweak is the name of the bands.
By default, the `LsstErrorModel` assumes that the LSST bands are named `u`, `g`, etc.
If instead, the bands in your catalog are named `lsst_u`, `lsst_g`, etc., then you can instantiate the error model with a rename dictionary:

```python
errModel = LsstErrorModel(renameDict={"u": "lsst_u", "g": "lsst_g", ...})
```

This tells `LsstErrorModel` to use all of the default parameters, but just change the names it gave to all of the bands.
If you are changing other dictionary-parameters at the same time (e.g. `nVisYr`, which sets the number of visits in each band per year), you can supply those parameters using *either* the new or old naming scheme!

### *Handling non-detections*

The other big thing you may want to change is how the error model identifies and handles non-detections.

The error model has a parameter named `sigLim`, which sets the limit for non-detections.
By default, `sigLim=0`, which means that only negative fluxes count as non-detections, however if you set `sigLim=1`, then any magnitudes beyond the 1-sigma limit in each band will count as a non-detection.
You can set `sigLim` to any non-negative float.

The `ndMode` parameter tells the error model how to handle the non-detections.
By default, `ndMode="flag"`, which means that the model will flag non-detections with the value set by `ndFlag`, which defaults to `np.inf`.
However, you can also set `ndMode="sigLim"`, in which case the model will set all non-detections to the n-sigma limits set by the `sigLim` parameter described in the previous paragraph.
Remember that `sigLim` also sets the detection threshold, so in effect, any galaxy magnitudes beyond the detection threshold will be set equal to the detection threshold.

One other option is provided by the `absFlux` parameter.
If `absFlux=True`, then the absolute value of all fluxes are taken before converting back to magnitudes.
If combined with `sigLim=0`, this means every galaxy will have an observed flux in every band.
This is useful if you do not want to worry about non-detections, but it results in a non-Gaussian error distribution for low-SNR sources.

### *Other error models*

In addition to `LsstErrorModel`, which comes with the LSST defaults, PhotErr includes `EuclidErrorModel` and `RomanErrorModel`, which come with the Euclid and Roman defaults, respectively.
Each of these models also have corresponding parameter objects: `EuclidErrorParams` and `RomanErrorParams`.

You can also start with the base error model, `ErrorModel`, which is not defaulted for any specific survey.
To instantiate `ErrorModel`, there are several required arguments that you must supply.
To see a list and explanation of these arguments, see the docstring for `ErrorParams`.

# Explanation of the error model

### *The point source model*

To derive the [Ivezic (2019)](https://arxiv.org/abs/0805.2366) error model, we start with the noise-to-signal ratio (NSR) for an object with photon count $C$ and background noise $N_0$ (which depends on seeing, read-out noise, etc.):

$$
NSR^2 = \frac{N_0^2 + C}{C^2}.
$$

If we define $C = C_5$ when $NSR = 1/5$, then we can solve for $N_0$ and write

$$
NSR^2 = \frac{1}{C_5} \left( \frac{C_5}{C} \right) + \left[ \left( \frac{1}{5} \right)^2 - \frac{1}{C_5} \right] \left( \frac{C_5}{C} \right)^2.
$$

Defining $x = \frac{C_5}{C} = 10 ^{(m - m_5) / 2.5}$ and $\gamma = \left( \frac{1}{5} \right)^2 - \frac{1}{C_5}$, we have

$$
NSR^2 = (0.04 - \gamma) x + \gamma x^2 ~~ (\text{mag}^2).
$$

In the high signal-to-noise ratio (SNR) limit, $NSR \ll 1$, and we can approximate

$$
\sigma_\text{rand} = 2.5 \log_{10}\left( 1 + NSR \right) \approx NSR.
$$

This approximation yields Equation 5 from [Ivezic (2019)](https://arxiv.org/abs/0805.2366).
In PhotErr, we do not make this approximation so that the error model generalizes to the low SNR regime.
In addition, while the high-SNR model assumes photometric errors are Gaussian in magnitude space, we model errors as Gaussian in flux space.
However, both of these high-SNR approximations can be restored with the keyword `highSNR=True`.

The LSST error model uses the parameters from [Ivezic (2019)](https://arxiv.org/abs/0805.2366).
The Euclid and Roman error models follow [Graham (2020)](https://arxiv.org/abs/2004.07885) in setting $\gamma = 0.04$, which is nearly identical to the values for Rubin (which are all $\sim 0.039$).

In addition to the random photometric error above, we include a system error of $\sigma_\text{sys} = 0.005$ which is added in quadrature to random error. Note that the system error can be changed using the keyword `sigmaSys`.

After adding photometric errors to the catalog, PhotErr recalculates the photometric error from the "observed" magnitudes.
This is so that the reported photometric errors do not provide a deterministic link back to the true magnitudes.
This behavior can be disabled by setting `decorrelate=False`.

### *The extended source model*

The [Ivezic (2019)](https://arxiv.org/abs/0805.2366) model calculates errors for point sources.
To model errors for extended sources, we use Equation 5 from [van den Busch (2020)](http://arxiv.org/abs/2007.01846):

$$
NSR_\text{ext} \propto
NSR_\text{pt} \sqrt{\frac{A_\text{ap}}{A_\text{psf}}},
$$

where $A_\text{ap}$ is the area of the source aperture, and $A_\text{psf}$ is the area of the PSF.
We set the proportionality constant to unity, so that when $A_\text{ap} \to A_\text{psf}$, we recover the error for a point source.

We include two different models for calculating the aperture area. The "auto" method from [van den Busch (2020)](http://arxiv.org/abs/2007.01846) calculates the semi-major and -minor axes of the aperture ( $a_\text{ap}$ and $b_\text{ap}$) from the semi-major and -minor axes of the galaxy ( $a_\text{gal}$ and $b_\text{gal}$, corresponding to half-light radii):

$$
a_\text{ap} = \sqrt{\sigma_\text{psf}^2 + (2.5 a_\text{gal})^2},
\quad
b_\text{ap} = \sqrt{\sigma_\text{psf}^2 + (2.5 b_\text{gal})^2},
$$

where $\sigma\_\text{psf} = \text{FWHM}\_\text{psf} / 2.355$ is the PSF standard deviation.
The formula for the area of an ellipse is then used to calculate the aperture area: $A_\text{ap} = \pi a_\text{ap} b_\text{ap}$.

The "gaap" method for extended sources ([Kuijken 2019](https://arxiv.org/abs/1902.11265)) is nearly identical, except that it adds a minimum aperture diameter in quadrature when calculating $a_\text{ap}$ and $b_\text{ap}$, and then clips aperture diameters above a certain maximum.

Calculating errors for extended sources requires columns in the galaxy catalog corresponding to the semi-major and -minor axes of the galaxies (with the length scale corresponding to the half-light radius).
You can set the names of these columns using the keywords `majorCol` and `minorCol`.

# Authors

[John Franklin Crenshaw](https://jfcrenshaw.github.io) \
[Ziang Yan](https://yanzastro.github.io)

[*Contributors guide*](CONTRIBUTING.md)
