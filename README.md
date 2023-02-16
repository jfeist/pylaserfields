# laserfields

`laserfields` is a library to describe the time-dependent electric fields of
a laser pulse. It implements the same pulse shapes and most of the features of
the [laserfields library](https://github.com/jfeist/laserfields) written in
Fortran. Please see the documentation of that library for the parameter
meanings, conventions used, etc. In particular, the "main" function
`make_laserfield(**kwargs...)` accepts the same parameters as the Fortran library
parameter files as keyword arguments, and returns an instance of a subtype of
the base class `LaserField` depending on the parameters. E.g., to create
a Gaussian pulse with a duration (defined as the FWHM of the intensity) of 6 fs,
a wavelength of 800 nm, a peak intensity of 1e14 W/cm^2, and with the peak at
time t=7fs, one should call
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

The "effective" duration of the pulse for n-photon processes can be obtained as
`lf.Teff(n_photon)`, which is the integral over the pulse intensity envelope to
the n-th power (i.e., electric field intensity envelope to the (2n)th power)
over the pulse, see, e.g., https://doi.org/10.1103/PhysRevA.77.043420 (Eq. 14).