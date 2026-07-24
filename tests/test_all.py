import pytest
import numpy as np
import laserfields
from laserfields.laserfields import LaserField
from laserfields import make_laserfield, GaussianLaserField, SinExpLaserField, LinearFlatTopLaserField, Linear2FlatTopLaserField, InterpolatingLaserField, LaserFieldCollection

general_args = dict(is_vecpot=True, E0=1.5, ω0=0.12, t0=500.0, chirp=0.0, ϕ0=0.8 * np.pi)
general_args_nonvec = dict(is_vecpot=False, E0=1.5, ω0=0.12, t0=500.0, chirp=0.0, ϕ0=0.8 * np.pi)
test_fields = [
    GaussianLaserField(**general_args, σ=100.0),
    SinExpLaserField(**general_args, T=800.0, exponent=2),
    SinExpLaserField(**general_args, T=800.0, exponent=4),
    SinExpLaserField(**general_args, T=800.0, exponent=7),
    LinearFlatTopLaserField(**general_args_nonvec, Tflat=400.0, Tramp=150),
    Linear2FlatTopLaserField(**general_args, Tflat=400.0, Tramp=150),
]


def test_general_args():
    expected_is_vecpot = [True, True, True, True, False, True]
    for i, lf in enumerate(test_fields):
        assert isinstance(lf, LaserField)
        assert lf.is_vecpot is expected_is_vecpot[i]
        assert lf.E0 == 1.5
        assert lf.ω0 == 0.12
        assert lf.t0 == 500
        assert lf.chirp == 0.0
        assert lf.ϕ0 == 0.8 * np.pi


def test_time_domain_posfreq_values():
    sample_t_shifts = (-0.2, 0.0, 0.3)
    expected_E_values = [
        [-0.4733721103340174, 1.213525491562421, 0.43939711942241655],
        [-0.4665774012671515, 1.213525491562421, 0.4560190763122897],
        [-0.4696176702845257, 1.213525491562421, 0.44856301902567197],
        [-0.47415631450844777, 1.213525491562421, 0.4374727545356417],
        [1.4265847744427302, 0.8816778784387098, -1.4265847744427274],
        [-0.4635254915624215, 1.213525491562421, 0.4635254915624302],
    ]
    expected_A_values = [
        [11.823200448244128, 7.347315653655915, -11.742442578919206],
        [11.868113281037923, 7.347315653655915, -11.843028666243946],
        [11.848054069403913, 7.347315653655915, -11.798022564282457],
        [11.818028803324419, 7.347315653655915, -11.730833895083071],
        [0.0, 0.0, 0.0],
        [11.888206453689419, 7.347315653655915, -11.888206453689396],
    ]
    expected_E_posfreq_values = [
        [complex(-0.2366860551670087, 0.7073805747087495), complex(0.6067627457812105, 0.4408389392193549), complex(0.21969855971120827, -0.7075431243057029)],
        [complex(-0.23328870063357574, 0.7114637065144673), complex(0.6067627457812105, 0.4408389392193549), complex(0.22800953815614486, -0.7115150383005721)],
        [complex(-0.23480883514226286, 0.7096391697345191), complex(0.6067627457812105, 0.4408389392193549), complex(0.22428150951283599, -0.7097408968808412)],
        [complex(-0.23707815725422388, 0.7069101152176173), complex(0.6067627457812105, 0.4408389392193549), complex(0.21873637726782086, -0.7070857016210783)],
        [complex(0.7132923872213651, 0.23176274578121075), complex(0.4408389392193549, -0.6067627457812105), complex(-0.7132923872213637, -0.23176274578121514)],
        [complex(-0.23176274578121075, 0.7132923872213651), complex(0.6067627457812105, 0.4408389392193549), complex(0.2317627457812151, -0.7132923872213638)],
    ]
    expected_A_posfreq_values = [
        [complex(5.911600224122064, 1.9207953490721235), complex(3.6736578268279576, -5.0563562148434205), complex(-5.871221289459603, -1.907675437887428)],
        [complex(5.934056640518961, 1.9280918810662835), complex(3.6736578268279576, -5.0563562148434205), complex(-5.921514333121973, -1.9240166383568338)],
        [complex(5.924027034701957, 1.924833064590886), complex(3.6736578268279576, -5.0563562148434205), complex(-5.899011282141228, -1.9167049538678569)],
        [complex(5.909014401662209, 1.9199551644239556), complex(3.6736578268279576, -5.0563562148434205), complex(-5.865416947541536, -1.9057894928745776)],
        [0j, 0j, 0j],
        [complex(5.944103226844709, 1.9313562148434231), complex(3.6736578268279576, -5.0563562148434205), complex(-5.944103226844698, -1.9313562148434595)],
    ]

    for i, lf in enumerate(test_fields):
        sample_times = [lf.t0 + shift * lf.TX for shift in sample_t_shifts]
        for j, t in enumerate(sample_times):
            assert np.allclose(lf.E(t), expected_E_values[i][j], atol=1e-12)
            assert np.allclose(lf.E_posfreq(t), expected_E_posfreq_values[i][j], atol=1e-12)
            assert np.allclose(lf.E(t), lf.E_posfreq(t) + lf.E_posfreq(t).conjugate(), atol=1e-12)
            if lf.is_vecpot:
                assert np.allclose(lf.A(t), expected_A_values[i][j], atol=1e-12)
                assert np.allclose(lf.A_posfreq(t), expected_A_posfreq_values[i][j], atol=1e-12)
                assert np.allclose(lf.A(t), lf.A_posfreq(t) + lf.A_posfreq(t).conjugate(), atol=1e-12)


def test_LaserFieldCollection():
    lfc = LaserFieldCollection(test_fields)
    lfc_vecpot = LaserFieldCollection([test_fields[i] for i in (0, 1, 2, 3, 5)])
    sample_t_shifts = (-0.2, 0.0, 0.3)
    assert isinstance(lfc, LaserFieldCollection)
    assert len(lfc.lfs) == 6
    assert lfc(500.0) == sum(lf(500.0) for lf in test_fields)
    assert lfc.E(300.0) == sum(lf.E(300.0) for lf in test_fields)
    assert lfc.E_posfreq(300.0) == sum(lf.E_posfreq(300.0) for lf in test_fields)
    for shift in sample_t_shifts:
        t = lfc.t0 + shift * lfc.TX
        assert np.allclose(lfc.E(t), lfc.E_posfreq(t) + lfc.E_posfreq(t).conjugate(), atol=1e-12)

    assert lfc_vecpot.A(300.0) == sum(lf.A(300.0) for lf in [test_fields[i] for i in (0, 1, 2, 3, 5)])
    assert lfc_vecpot.A_posfreq(300.0) == sum(lf.A_posfreq(300.0) for lf in [test_fields[i] for i in (0, 1, 2, 3, 5)])
    for shift in sample_t_shifts:
        t = lfc_vecpot.t0 + shift * lfc_vecpot.TX
        assert np.allclose(lfc_vecpot.A(t), lfc_vecpot.A_posfreq(t) + lfc_vecpot.A_posfreq(t).conjugate(), atol=1e-12)

    assert lfc.E_fourier(1.0) == sum(lf.E_fourier(1.0) for lf in test_fields)
    assert lfc_vecpot.A_fourier(1.0) == sum(lf.A_fourier(1.0) for lf in [test_fields[i] for i in (0, 1, 2, 3, 5)])
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
    with pytest.raises(ValueError):
        lf.E_posfreq(350.0)
    with pytest.raises(ValueError):
        lf.A_posfreq(350.0)


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
    with pytest.raises(ValueError):
        lf.E_posfreq(350.0)
    with pytest.raises(ValueError):
        lf.A_posfreq(350.0)


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

    with pytest.raises(ValueError):
        make_laserfield(form="linear", is_vecpot=True, duration=1000.0, rampon=100.0, E0=1.0, omega=1.0, t0=0.0)
    with pytest.raises(ValueError):
        make_laserfield(form="sin_exp", is_vecpot=True, duration=1000.0, form_exponent=1.0, E0=1.0, omega=1.0, t0=0.0)


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
            is_vecpot = False if form == "linear" else True
            lf = make_laserfield(form=form, is_vecpot=is_vecpot, duration=1000.0, rampon=100.0, E0=1.0, omega=1.0, t0=0.0)
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
            LinearFlatTopLaserField(**{**general_args, "is_vecpot": False}, Tflat=400.0, Tramp=150),
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
