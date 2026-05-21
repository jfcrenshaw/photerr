<div align="center">

[![build](https://github.com/jfcrenshaw/photerr/actions/workflows/main.yml/badge.svg)](https://github.com/jfcrenshaw/photerr/actions/workflows/main.yml)
[![Codecov](https://img.shields.io/codecov/c/github/jfcrenshaw/photerr?label=codecov&logo=codecov)](https://app.codecov.io/gh/jfcrenshaw/photerr)
[![PyPI](https://img.shields.io/pypi/v/photerr?color=blue&label=PyPI)](https://pypi.org/project/photerr/)

</div>

# PhotErr

PhotErr is a photometric error model for astronomical imaging surveys.
It implements a generalization of the high-SNR point-source error model from [Ivezic (2019)](https://arxiv.org/abs/0805.2366) that is more accurate in the low SNR regime and includes errors for extended sources, using the models from [van den Busch (2020)](http://arxiv.org/abs/2007.01846) and [Kuijken (2019)](https://arxiv.org/abs/1902.11265).

PhotErr currently includes photometric error models for the Vera C. Rubin Observatory Legacy Survey of Space and Time (LSST), as well as the Euclid and Nancy Grace Roman space telescopes.

If you use this package in your research, please cite the following paper

```bibtex
@ARTICLE{2024AJ....168...80C,
       author = {{Crenshaw}, John Franklin and {Kalmbach}, J. Bryce and {Gagliano}, Alexander and {Yan}, Ziang and {Connolly}, Andrew J. and {Malz}, Alex I. and {Schmidt}, Samuel J. and {The LSST Dark Energy Science Collaboration}},
        title = "{Probabilistic Forward Modeling of Galaxy Catalogs with Normalizing Flows}",
      journal = {\aj},
     keywords = {Neural networks, Galaxy photometry, Surveys, Computational methods, 1933, 611, 1671, 1965, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2024,
        month = aug,
       volume = {168},
       number = {2},
          eid = {80},
        pages = {80},
          doi = {10.3847/1538-3881/ad54bf},
archivePrefix = {arXiv},
       eprint = {2405.04740},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024AJ....168...80C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

# Getting started

PhotErr is available on PyPI and can be installed with pip:

```bash
pip install photerr
```

Note that PhotErr requires Python >= 3.10.

Once installed, you can import the error models. For example, to use the default LSST error model,

```python
from photerr import LsstErrorModel
errModel = LsstErrorModel()
catalog_with_errors = errModel(catalog, random_state=42)
```

The error model expects an input catalog in the form of a pandas DataFrame with true magnitudes, and it returns another DataFrame containing observed magnitudes and photometric errors.
Any extraneous columns in the DataFrame (e.g. a redshift column), remain in the new DataFrame - their presence does not effect the error model.

*If compatibility with Astropy Tables, Ordered Dictionaries, etc., would be useful to you, let me know!*

You can also calculate limiting magnitudes:

```python
errModel.getLimitingMags() # coadded point-source 5-sigma limits
errModel.getLimitingMags(nSigma=1, coadded=False) # single-image point-source 1-sigma limits
```

# Tweaking the error model

There are many parameters you can tweak to fine tune the error model.
To see all available parameters, check the docstring of either the error model or parameters object.
For example,

```python
from photerr import LsstErrorModel
help(LsstErrorModel)
```

All model parameters can be overridden using keyword arguments to the error model constructor.
Below, we explain in detail a few of the more commonly tweaked parameters.

### *Changing the observing duration*

The example above uses the default settings for the LSST model, which includes 10 years of observing time.
If instead you want to calculate errors for LSST year 1, you can pass the `nYrObs` argument to the constructor:

```python
errModel = LsstErrorModel(nYrObs=1)
```

### *Directly setting limiting magnitudes*

By default, PhotErr tries to use the provided information to calculate limiting magnitudes for you.
If you would like to directly supply your own $5\sigma$ limits, you can do so using the `m5` parameter.
Note PhotErr assumes these are single-visit point-source limiting magnitudes.
If you want to supply coadded depths, you should also set `nYrObs=1` and `nVisYr=1`, so the calculated coadded depths are equal to those you provided.

### *Changing the band names*

Another parameter you might want to tweak is the name of the bands.
By default, the `LsstErrorModel` assumes the LSST bands are named `u`, `g`, `r`, etc.
If instead, the bands in your catalog are named `lsst_u`, `lsst_g`, `lsst_r`, etc., you can instantiate the error model with a rename dictionary:

```python
errModel = LsstErrorModel(renameDict={"u": "lsst_u", "g": "lsst_g", ...})
```

This tells `LsstErrorModel` to use all of the default parameters, but just change the names it gave to all of the bands.
If you are changing other dictionary-parameters at the same time (e.g. `nVisYr`, which sets the number of visits in each band per year), you can supply those parameters using *either* the new or old naming scheme!

### *Handling non-detections*

The other big thing you may want to change is how the error model identifies and handles non-detections.

The error model has a parameter named `sigLim`, which sets the SNR threshold for non-detections.
By default `sigLim=0`, meaning no SNR threshold is applied.
If you set `sigLim=1`, any source with SNR below 1 in a given band will be treated as a non-detection in that band.
You can set `sigLim` to any non-negative float.

When using the default `outputType="pogson"`, sources with negative observed fluxes are always treated as non-detections regardless of `sigLim`, because negative fluxes cannot be represented as Pogson magnitudes.

The `ndMode` parameter tells the error model how to handle the non-detections.
By default `ndMode="flag"`, which means the model will flag non-detections with the value set by `ndFlag`, which defaults to `np.inf`.
However, you can also set `ndMode="sigLim"`, in which case the model will set all non-detections to the n-sigma limits set by the `sigLim` parameter described in the previous paragraph.
Remember that `sigLim` also sets the detection threshold, so in effect, any galaxy magnitudes beyond the detection threshold will be set equal to the detection threshold.

One other option is provided by the `absFlux` parameter.
If `absFlux=True`, the absolute value of all fluxes are taken before converting back to magnitudes.
If combined with `sigLim=0`, this means every galaxy will have an observed flux in every band.
This is useful if you do not want to worry about non-detections, but it results in a non-Gaussian error distribution for the flux of low-SNR sources.

The cleanest way to avoid non-detections while preserving the correct (Gaussian-in-flux) error distribution is to use `outputType="maggy"` or `outputType="asinh"` instead.
In these modes, negative observed fluxes are **not** flagged as non-detections — they are valid measurements, as they should be for very faint sources near the noise floor.
No change to `sigLim` is needed: the default `sigLim=0` already means no detection threshold is applied, so all negative-flux sources are preserved.
See *[Using alternative magnitude/flux systems](#using-alternative-magnitudeflux-systems)* below.

### *Errors for extended sources*

PhotErr can be used to calculate errors for extended sources as well.
You just have to pass `extendedType="auto"` or `extendedType="gaap"` to the constructor (see explanation below for the differences in these models).
PhotErr will then look for columns in the input DataFrame that correspond to the semi-major and -minor axes of the objects, corresponding to half-light radii in arcseconds.
By default it looks for these in columns titled "major" and "minor", but you can change the names of these columns using the `majorCol` and `minorCol` keywords.

You can also calculate limiting magnitudes for apertures of a given size by passing the `aperture` keyword to `errModel.getLimitingMags()`

### *Scaling the errors*

If you want to scale up or scale down the errors in any band(s), you can use the keyword `scale`.
For example, `LsstErrorModel(scale={"u": 2, "y": 2})` will have all the same properties of the default error model, except the errors in the `u` and `y` bands will be doubled.
This allows you to answer questions like "what happens to my science if the `u` band errors are doubled."

Note it is the flux error that is doubled.
This also only scales the band-specific error.
The band-independent systematic error floor, `sigmaSys` is still the same, and so at high-SNR near the systematic floor the errors won't scale as you expect.

### *Using alternative magnitude/flux systems*

By default, PhotErr expects input magnitudes and returns output magnitudes in standard Pogson magnitudes (i.e., $m = -2.5 \log_{10}(f/f_0)$).
You can change this with the `inputType` and `outputType` parameters, each of which accepts one of three values:

- `"pogson"` (default) — standard Pogson magnitudes.
- `"maggy"` — linear fluxes in *maggies*, where a source with magnitude $m = 0$ has flux $f = 1$ maggy (i.e. $f = 10^{-m/2.5}$).
- `"asinh"` — *asinh magnitudes* (also called *luptitudes*), defined by [Lupton et al. (1999)](https://ui.adsabs.harvard.edu/abs/1999AJ....118.1406L/abstract) as

$$\mu = -\frac{2.5}{\ln 10}\left[\text{arcsinh}\!\left(\frac{f}{2b}\right) + \ln b\right],$$

where $f$ is the flux in maggies and $b$ is a per-band softening parameter in maggies.

`inputType` and `outputType` are independent: you can, for example, read in asinh magnitudes and get back maggies, or read in Pogson magnitudes and get back asinh magnitudes.

For example,

```python
# Read Pogson mags, return linear fluxes in maggies
errModel = LsstErrorModel(outputType="maggy")
obs = errModel(catalog_pogson, random_state=42)  # obs band columns are in maggies

# Read and return asinh magnitudes (luptitudes)
errModel = LsstErrorModel(inputType="asinh", outputType="asinh")
obs = errModel(catalog_luptitude, random_state=42)
```

**Softening parameter** `asinhB`

The asinh magnitude formula requires a per-band softening parameter $b$ (in maggies) that controls the flux scale at which the transition from logarithmic to linear behavior occurs.
By default, $b$ is set to the coadded 1$\sigma$ limiting flux in each band, which places the softening at the survey noise floor — a natural and commonly used choice.
You can override this per-band or globally with the `asinhB` parameter.
Note that if your data is already in asinh magnitudes (i.e., you set `inputType="asinh"`) make sure you set `asinhB` equal to the values used in the creation of your catalog!


**Negative fluxes and interaction with `sigLim` / `ndMode`**

The key advantage of maggies and asinh magnitudes over Pogson magnitudes is that they are well-defined for negative observed fluxes.
For very faint sources near the noise floor, the observed flux is drawn from a distribution centred on the true (positive) flux with standard deviation $\sigma_f = f_\text{true} \times \text{NSR}$.
When $\text{NSR} \gtrsim 1$, a substantial fraction of draws will be negative.
If you wish to preserve the full Gaussian noise distribution at the low-SNR end, use either `inputType="maggy"` or `inputType="asinh"`, and leave `sigLim=0` and `ndMode="flag"` (these are the defaults).

### *Other error models*

In addition to `LsstErrorModel`, which comes with the LSST defaults, PhotErr includes

- `EuclidWideErrorModel` (also aliased as `EuclidErrorModel`)
- `EuclidDeepErrorModel`
- `RomanWideErrorModel`
- `RomanMediumErrorModel` (also aliased as `RomanErrorModel`)
- `RomanDeepErrorModel`
- `RomanUltraDeepErrorModel`

Each of these models also have corresponding parameter object, e.g. `RomanErrorParams`.

You can also start with the base error model, `ErrorModel`, which is not defaulted for any specific survey.
To instantiate `ErrorModel`, there are several required arguments that you must supply.
To see a list and explanation of these arguments, see the docstring for `ErrorModel`.
However, the easiest way to create a new model is to supply `nYrObs`, `nVisYr`, `gamma`, and `m5`.
You might need to fit `gamma` to match the expected errors, however a good default guess is `0.04`.

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
NSR^2 = (0.04 - \gamma) x + \gamma x^2.
$$

In the high signal-to-noise ratio (SNR) limit, $NSR \ll 1$, and we can approximate

$$
\sigma_\text{rand} = 2.5 \log_{10}\left( 1 + NSR \right) \approx NSR.
$$

This approximation yields Equation 5 from [Ivezic (2019)](https://arxiv.org/abs/0805.2366).
In PhotErr, we do not make this approximation so that the error model generalizes to the low SNR regime.
In addition, while the high-SNR model assumes photometric errors are Gaussian in magnitude space, we model errors as Gaussian in flux space.
However, both of these high-SNR approximations can be restored with the keyword `highSNR=True`.

The LSST error model uses parameters from [Ivezic (2019)](https://arxiv.org/abs/0805.2366), [Bianco 2022](https://pstn-054.lsst.io), and from [this Rubin systems engineering notebook](https://github.com/lsst-pst/syseng_throughputs/blob/main/notebooks/EvalReadnoise.ipynb).
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
Note the PSF FWHM is assumed to scale like $\text{airmass}^{0.6}$.
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
