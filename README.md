# laserfields

`laserfields` is a python library to describe the time-dependent electric fields
of a laser pulse, and can be installed with `pip install laserfields` or `conda
-c conda-forge install laserfields`. It implements the same pulse shapes and
most of the features of the [laserfields
library](https://github.com/jfeist/laserfields) written in Fortran (and as the
Julia variant [LaserFields.jl](https://github.com/jfeist/LaserFields.jl)),
please see the documentation of that library for the parameter meanings,
conventions used, etc. In particular, the "main" function
`make_laserfield(**kwargs...)` accepts the same parameters as the Fortran
library parameter files as keyword arguments, and returns an instance of a
subtype of the base class `LaserField` depending on the parameters. E.g., to
create a Gaussian pulse with a duration (defined as the FWHM of the intensity)
of 6 fs, a wavelength of 800 nm, a peak intensity of 1e14 W/cm^2, and with the
peak at time t=7fs, one should call
```python
lf = make_laserfield(form="gaussianI", is_vecpot=true, lambda_nm=800,
                      intensity_Wcm2=1e16, duration_as=6000, peak_time_as=7000)
```

Given a `LaserField` instance `lf`, the functions `lf.E(t)`,
`lf.E_fourier(ω)`, `lf.A(t)`, and `lf.A_fourier(ω)` can be used to obtain,
respectively, the electric field as a function of time, its Fourier transform
(implemented for most pulse shapes), the vector potential as a function of time,
and its Fourier transform. Calling the instance as a function, `lf(t)` returns
the electric field, i.e., is equivalent to `lf.E(t)`. The notebooks in the
`examples` folder show some ways to use the library, including how to define a
set of fields through a YAML configuration file.

In addition to the pulses described by each `LaserField` instance, the library
also implements a `LaserFieldCollection` class that can be used to combine
multiple fields into a single effective one (i.e., the sum of the individual
ones). It is also a `LaserField` instance and supports much of the same
interface. Note that some of the parameters it contains are just "best-effort"
values and may not be fully meaningful for the combined field - e.g., for the
carrier frequency `lf.ω0`, it returns the highest value in the collection, to
support use cases where this is used to define maximum time step in a numerical
propagation, or the maximum frequency evaluated in a Fourier transform.

The "effective" duration of the pulse for n-photon processes can be obtained as
`lf.Teff(n_photon)`, which is the integral over the pulse intensity envelope to
the n-th power (i.e., electric field envelope to the (2n)th power)
over the pulse, see, e.g., https://doi.org/10.1103/PhysRevA.77.043420 (Eq. 14).
