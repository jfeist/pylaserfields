import pytest
import numpy as np
import laserfields
from laserfields.laserfields import LaserField
from laserfields import make_laserfield, GaussianLaserField, SinExpLaserField, LinearFlatTopLaserField, Linear2FlatTopLaserField, InterpolatingLaserField, LaserFieldCollection

general_args = dict(is_vecpot=True, E0=1.5, ω0=0.12, t0=500.0, chirp=0.0, ϕ0=0.8 * np.pi)
test_fields = [
    GaussianLaserField(**general_args, σ=100.0),
    SinExpLaserField(**general_args, T=800.0, exponent=2),
    SinExpLaserField(**general_args, T=800.0, exponent=4),
    SinExpLaserField(**general_args, T=800.0, exponent=7),
    LinearFlatTopLaserField(**general_args, Tflat=400.0, Tramp=150),
    Linear2FlatTopLaserField(**general_args, Tflat=400.0, Tramp=150),
]


def test_general_args():
    for lf in test_fields:
        assert isinstance(lf, LaserField)
        assert lf.is_vecpot is True
        assert lf.E0 == 1.5
        assert lf.ω0 == 0.12
        assert lf.t0 == 500
        assert lf.chirp == 0.0
        assert lf.ϕ0 == 0.8 * np.pi


def test_LaserFieldCollection():
    lfc = LaserFieldCollection(test_fields)
    assert isinstance(lfc, LaserFieldCollection)
    assert len(lfc.lfs) == 6
    assert lfc(500.0) == sum(lf(500.0) for lf in test_fields)
    assert lfc.E(300.0) == sum(lf.E(300.0) for lf in test_fields)
    assert lfc.A(300.0) == sum(lf.A(300.0) for lf in test_fields)
    assert lfc.E_fourier(1.0) == sum(lf.E_fourier(1.0) for lf in test_fields)
    assert lfc.A_fourier(1.0) == sum(lf.A_fourier(1.0) for lf in test_fields)
    assert lfc.start_time == np.min([lf.start_time for lf in test_fields])
    assert lfc.end_time == np.max([lf.end_time for lf in test_fields])


def test_readin_vecpot():
    lf = InterpolatingLaserField(datafile="tests/laserdat.dat", is_vecpot=True)
    assert lf.is_vecpot is True
    assert np.isclose(lf.E0, 0.15985646054964597)
    assert np.isclose(lf.ω0, 0.160976529593676)
    assert np.isclose(lf.t0, 353.38806594930224)
    assert lf.duration == 700.0
    assert lf.ϕ0 == 0.0
    assert lf.chirp == 0.0
    assert lf.datafile == "tests/laserdat.dat"
    assert lf.start_time == 0.0
    assert lf.end_time == 700.0


def test_readin_efield():
    lf = InterpolatingLaserField(datafile="tests/laserdat.dat", is_vecpot=False)
    assert lf.is_vecpot is False
    assert np.isclose(lf.E0, 0.9968360392353086)
    assert np.isclose(lf.ω0, 0.1600000889708898)
    assert np.isclose(lf.t0, 343.6504511282523)
    assert lf.duration == 700.0
    assert lf.ϕ0 == 0.0
    assert lf.chirp == 0.0
    assert lf.datafile == "tests/laserdat.dat"
    assert lf.start_time == 0.0
    assert lf.end_time == 700.0


def test_make_laserfield():
    lf = make_laserfield(form="gaussianI", is_vecpot=True, phase_pi=1, duration_as=100.0, peak_time_as=400, intensity_Wcm2=1e14, lambda_nm=12.0, linear_chirp_rate_w0as=0.0)
    assert isinstance(lf, GaussianLaserField)
    assert lf.is_vecpot is True
    assert lf.σ == 100.0 * laserfields.au_as / np.sqrt(np.log(16.0))
    assert lf.t0 == 400.0 * laserfields.au_as
    assert lf(lf.t0) == lf.E0
    assert lf.ϕ0 == np.pi

    with pytest.raises(ValueError):
        make_laserfield(form="gaussianI", is_vecpot=True, phase_pi=0.5, duration=10.0, duration_as=100.0, peak_time_as=400, intensity_Wcm2=1e14, lambda_nm=12.0, linear_chirp_rate_w0as=0.0)
    with pytest.raises(ValueError):
        make_laserfield(form="gaussianI", is_vecpot=True, phase_pi=0.5, duration_as=100.0, peak_time=0.0, peak_time_as=400, intensity_Wcm2=1e14, lambda_nm=12.0, linear_chirp_rate_w0as=0.0)
    with pytest.raises(ValueError):
        make_laserfield(form="gaussianI", is_vecpot=True, phase_pi=0.5, duration_as=100.0, peak_time_as=400, E0=0.3, intensity_Wcm2=1e14, lambda_nm=12.0, linear_chirp_rate_w0as=0.0)

    with pytest.raises(ValueError):
        make_laserfield(form="gaussianI", is_vecpot=True, phase_pi=0.5, peak_time_as=400, intensity_Wcm2=1e14, lambda_nm=12.0, linear_chirp_rate_w0as=0.0)
    with pytest.raises(ValueError):
        make_laserfield(form="gaussianI", is_vecpot=True, phase_pi=0.5, duration_as=100.0, intensity_Wcm2=1e14, lambda_nm=12.0, linear_chirp_rate_w0as=0.0)
    with pytest.raises(ValueError):
        make_laserfield(form="gaussianI", is_vecpot=True, phase_pi=0.5, duration_as=100.0, peak_time_as=400, lambda_nm=12.0, linear_chirp_rate_w0as=0.0)


def test_Teff():
    refTs = dict(
        gaussianI=[1064.4670194312, 752.69184778925, 614.5703202121, 532.23350971561, 476.04412305096],
        gaussianF=[752.69184778925, 532.23350971561, 434.56684093796, 376.34592389463, 336.61402755334],
        sin2=[375, 273.4375, 225.5859375, 196.38061523438, 176.19705200195],
        sin4=[273.4375, 196.38061523438, 161.18025779724, 139.94993409142, 125.37068761958],
        linear=[1066.6666666667, 1040, 1028.5714285714, 1022.2222222222, 1018.1818181818],
        linear2=[1075, 1054.6875, 1045.1171875, 1039.2761230469, 1035.2394104004],
    )
    for form, Teffs in refTs.items():
        for n_photon, T in enumerate(Teffs, start=1):
            lf = make_laserfield(form=form, is_vecpot=True, duration=1000.0, rampon=100.0, E0=1.0, omega=1.0, t0=0.0)
            assert np.isclose(lf.Teff(n_photon), T)


def test_fourier():
    # Test the Fourier transform of the laser fields
    # Compare the analytical Fourier transform with the numerical one (computed using FFT)
    for chirp in -0.0011, -0.0009, -1e-3, -1e-20, 0, 1e-20, 1e-3, 0.0009, 0.0011:
        general_args = dict(is_vecpot=True, E0=1.5, ω0=0.12, t0=500.0, chirp=chirp, ϕ0=0.8 * np.pi)
        for lf in [
            GaussianLaserField(**general_args, σ=100.0),
            SinExpLaserField(**general_args, T=100.0, exponent=2),
            SinExpLaserField(**general_args, T=100.0, exponent=4),
            SinExpLaserField(**general_args, T=100.0, exponent=7),
            LinearFlatTopLaserField(**general_args, Tflat=400.0, Tramp=150),
            Linear2FlatTopLaserField(**general_args, Tflat=400.0, Tramp=150),
        ]:
            if isinstance(lf, (LinearFlatTopLaserField, Linear2FlatTopLaserField)) and chirp != 0:
                # Skip the test for LinearFlatTopLaserField with non-zero chirp
                # because the analytical Fourier transform is not implemented for this case
                continue
            T = lf.end_time - lf.start_time
            t0 = lf.start_time - 5 * T
            t1 = lf.end_time + 5 * T
            dt = lf.TX / 100
            ts = np.arange(t0, t1, dt)
            ωs = 2 * np.pi * np.fft.fftfreq(len(ts), dt)
            Eω = lf.E_fourier(ωs)
            Eω2 = np.fft.fft(lf(ts)) * dt / np.sqrt(2 * np.pi)
            # FFT acts as if ts[0] was t=0, shift to the correct value
            Eω2 *= np.exp(-1j * ts[0] * ωs)

            atol = 0.02 if isinstance(lf, LinearFlatTopLaserField) else 1e-3
            assert np.allclose(Eω, Eω2, atol=atol), f"Failed for {lf.__class__.__name__} with chirp {chirp}"
