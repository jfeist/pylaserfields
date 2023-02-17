import numpy as np
from numpy import exp, log, sin, cos, sqrt, pi as π
from dataclasses import dataclass
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import gamma, erf
from numba import njit, vectorize

au_as   = 1/24.188843265     # attosecond in a.u.
au_wcm2toel2 = 1/3.509338e16 # W/cm^2 in a.u. for electric field squared
au_wcm2 = 1.55369e-16        # W/cm^2 in a.u
au_m    = 1/5.291772098e-11  # m in a.u.
au_cm   = 1/5.291772098e-9   # cm in a.u.
au_nm   = 1/5.291772098e-2   # nm in a.u.
au_c    = 137.03599911       # c (speed of light) in a.u. == 1/alpha
au_eV   = 1/27.2113845       # eV in a.u.
au_m_He = 7294.2995365       # m of He-nucleus in a.u.
au_m_n  = 1838.6836605       # m of neutron in a.u.

GAUSSIAN_TIME_CUTOFF_SIGMA = 3.5*sqrt(log(256))

@dataclass
class LaserField:
    is_vecpot: bool
    E0: float
    ω0: float
    t0: float
    chirp: float
    ϕ0: float

    TX = property(lambda self: 2*π/self.ω0)

    def __call__(self,t):
        return self.E(t)

    def E(self,t):
        tr = np.asarray(t) - self.t0
        env, envpr = self._envelope(tr)
        phit = self.ϕ0 + self.ω0*tr + self.chirp*tr**2
        osc = sin(phit)

        if self.is_vecpot:
            # d(phi(t))/dt = self.ω0 + 2*self.chirp*tr
            oscpr = (self.ω0 + 2*self.chirp*tr)*cos(phit)
            return -(env * oscpr + envpr * osc) / self.ω0
        else:  # describes electric field directly
            return env * osc

    def A(self,t):
        if not self.is_vecpot:
            raise ValueError('laser field is not given as a vector potential, cannot get A(t) analytically!')

        tr = np.asarray(t) - self.t0
        env, envpr = self._envelope(tr)
        osc = sin(self.ϕ0 + self.ω0*tr + self.chirp*tr**2)
        # Divide out derivative of oscillation to ensure peak amplitude of E0 for electric field
        return env*osc / self.ω0

    def envelope(self,t):
        return self._envelope(np.asarray(t) - self.t0)[0]

    def _envelope(self,tr):
        raise NotImplementedError()

    """return the fourier transform of the envelope of the laser field.
    we write the whole pulse as
    f(t) = (env(t) exp(i*(phi0 + w0*tp + chirp*tp**2)) + c.c. ) / (2*IU), where tp = t-tpeak
    for the fourier transform of the envelope, we include the chirp term
    exp(i chirp (t-tpeak)**2) in the envelope, so that its fourier transform is a complex function.
    however, for unchirped pulses, the result will be purely real!

    for the various calculations, see chirped_fourier.nb in the mathematica directory."""
    def _envelope_fourier(self,omega):
        raise NotImplementedError()

    def E_fourier(self,omega):
        # analytically determine the fourier transform of the defined laser fields
        # determined as Int exp(-i*omega*t) E(t) dt

        # with tp = t-tpeak, the whole pulse is
        # f(t) =  env(t) sin    (phi0 + w0*tp + chirp*tp**2)
        #      = (env(t) exp(IU*(phi0 + w0*tp + chirp*tp**2)) - c.c. ) / (2*IU)
        # for the fourier transform, we include the chirp term exp(i chirp tp**2) in the envelope.
        # this part is transformed in lf_envelope_fourier.
        # exp(IU*phi0) is just a constant prefactor, and the linear phase w0*tp just gives a shift in frequency,
        # F[f(t) exp(IU w0 t)](w) = F[f(t)](w-w0)
        # complex conjugation of the transformed function gives complex conjugation + reversal of the argument in the transform, so
        # F[conjg(f(t) exp(IU w0 t))](w) = conjg(F[f(t) exp(IU w0 t)](-w)) = conjg(F[f(t)](-w-w0))
        ELFT = (   self._envelope_fourier( omega - self.ω0) * exp(1j*self.ϕ0)
                - (self._envelope_fourier(-omega - self.ω0) * exp(1j*self.ϕ0)).conj()) / (2j)

        # the fourier transform of the part was determined as if it was centered around t=0
        # shift in time now -- just adds a phase exp(-IU*omega*t0), as F[f(t-a)] = exp(-IU*omega*a) F[f(t)]
        ELFT *= exp(-1j*omega*self.t0)

        if self.is_vecpot:
            # if this laser field was defined as a vector potential, we need to multiply
            # with -IU*omega to get the fourier transform of the electric field, E=-dA/dt
            # F[-dA/dt] = -iw F[A]
            # in addition, we need to take into account that A0 = E0 / lf%omega
            ELFT *= -1j * omega / self.ω0

        return ELFT

    def A_fourier(self,omega):
        return self.E_fourier(omega) / (-1j*omega)

    @property
    def start_time(self):
        raise NotImplementedError()

    @property
    def end_time(self):
        raise NotImplementedError()

    """returns the "effective duration" of a laser field for n-photon processes
    the values for T_eff are calculated according to
    I_0^n * T_eff = \Int_0^T I(t)^n dt = \Int_0^T envelope(t)^(2n) dt"""
    def Teff(self,n_photon):
        raise NotImplementedError()

@dataclass
class GaussianLaserField(LaserField):
    σ: float

    def _envelope(self,tr):
        env   = self.E0 * exp(-tr**2/(2*self.σ**2))
        envpr = -env * tr/self.σ**2
        return env,envpr
    def _envelope_fourier(self,omega):
        # F[exp(-z*t**2)] = exp(-w**2/4*z)/sqrt(2*z) (for real(z)>0)
        z = 0.5/self.σ**2 - 1j*self.chirp
        return self.E0 * exp(-omega**2/(4*z)) / sqrt(2*z)
    start_time = property(lambda self: self.t0 - GAUSSIAN_TIME_CUTOFF_SIGMA*self.σ)
    end_time   = property(lambda self: self.t0 + GAUSSIAN_TIME_CUTOFF_SIGMA*self.σ)
    Teff = lambda self,n_photon: self.σ * sqrt(π/n_photon)

def expiatbt2_intT(a,b,T):
    # returns the result of the integral Int(exp(i*(a*t+b*t**2)),{t,-T/2,T/2}) / sqrt(2*pi)
    zz1 = (1+1j)/4
    z34 = (-1.+1j)/sqrt(2) # == (-1)**(3/4)
    b = complex(b) # we want to take the square root and b might be negative
    res = erf(z34*(a-b*T)/sqrt(4*b)) - erf(z34*(a+b*T)/sqrt(4*b))
    res = res * zz1 / sqrt(b) * exp(-1j*a**2/(4*b))
    # this is surprisingly not given by mathematica - not sure yet why it misses it,
    # but it's necessary for agreement with the numerical fourier transform
    return res * np.sign(b)

@dataclass
class SinExpLaserField(LaserField):
    T: float
    exponent: float

    def _envelope(self,tr):
        if not hasattr(self,'_jit_envelope'):
            @vectorize(nopython=True)
            def _jit_envelope(tr,E0,T,exponent,getprime):
                trel = tr/T
                if abs(trel) > 0.5:
                    return 0.
                if getprime:
                    return -E0 * sin(π*trel) * exponent * cos(π*trel)**(exponent-1) * π/T
                else:
                    return E0 * cos(π*trel)**exponent
            self._jit_envelope = _jit_envelope
        env   = self._jit_envelope(tr,self.E0,self.T,self.exponent,False)
        envpr = self._jit_envelope(tr,self.E0,self.T,self.exponent,True)
        return env,envpr
    def _envelope_fourier(self,omega):
        if self.exponent == 2:
            if self.chirp == 0:
                # the expression with chirp can not be evaluated with chirp == 0, so we take this as a special case
                return self.E0 * sqrt(8*π**3) * np.sinc(omega*self.T/(2*π))/(8*π**2/self.T - 2*omega**2*self.T)
            else:
                # now we use that cos(pi*t/T)**2 * exp(i*c*t**2) can be written as 0.5 exp(i*c*t**2) + 0.25 exp(i*c*t**2 - 2*i*pi*t/T) + 0.25 exp(i*c*t**2 + 2*i*pi*t/T)
                # the integral of exp(IU*(a*t+b*t**2)) from t=-T/2 to t=T/2 can be calculated analytically and is implemented in the function below
                # the arguments are a={-omega, -2*pi/T-omega, 2*pi/T-omega} and b=chirp
                wd = 2*π/self.T
                return self.E0 * (expiatbt2_intT(    - omega, self.chirp, self.T)/2 +
                                  expiatbt2_intT(-wd - omega, self.chirp, self.T)/4 +
                                  expiatbt2_intT( wd - omega, self.chirp, self.T)/4)
        elif self.exponent == 4:
            if self.chirp == 0:
                # the expression with chirp can not be evaluated with chirp == 0, so we take this as a special case
                return self.E0 * 24 * (sqrt(2*π**7) * np.sinc(omega*self.T/(2*π)) /
                                       (128*π**4/self.T - 40*π**2*omega**2*self.T + 2*omega**4*self.T**3))
            else:
                # now we use that cos(pi*t/T)**4 * exp(i*c*t**2) can be written as
                # (0.375 exp(i*c*t**2) + 0.25 exp(i*c*t**2 - 2*i*pi*t/T) + 0.25 exp(i*c*t**2 + 2*i*pi*t/T) +
                #  0.0625 exp(i*c*t**2 - 4*i*pi*t/T) + 0.0625 exp(i*c*t**2 + 4*i*pi*t/T))
                # the integral of exp(IU*(a*t+b*t**2)) from t=-T/2 to t=T/2 can be calculated analytically and is implemented in the function below
                # the arguments are a={-omega, -2*pi/T-omega, 2*pi/T-omega, -4*pi/T-omega, 4*pi/T-omega} and b=chirp
                wd = 2*π/self.T
                return self.E0 * (expiatbt2_intT(      - omega, self.chirp, self.T)*0.375  +
                                  expiatbt2_intT(  -wd - omega, self.chirp, self.T)*0.25   +
                                  expiatbt2_intT(   wd - omega, self.chirp, self.T)*0.25   +
                                  expiatbt2_intT(-2*wd - omega, self.chirp, self.T)*0.0625 +
                                  expiatbt2_intT( 2*wd - omega, self.chirp, self.T)*0.0625)
        else:
            if self.chirp != 0 or not float(self.exponent).is_integer():
                raise NotImplementedError('sin_exp fourier transform with exponent != 2 or 4 only implemented for integer exponents and unchirped pulses')
            x = 0.5*(omega*self.T/π - self.exponent)
            return self.E0 * self.T * gamma(self.exponent+1)*gamma(x)*sin(π*x)/(sqrt(2**(2*self.exponent+1) * π**3) * gamma(x+self.exponent+1))

    start_time = property(lambda self: self.t0 - self.T/2)
    end_time   = property(lambda self: self.t0 + self.T/2)
    Teff = lambda self,n_photon: self.T * gamma(0.5 + n_photon*self.exponent) / (sqrt(π)*gamma(1 + n_photon*self.exponent))

@dataclass
class FlatTopLaserField(LaserField):
    Tflat: float
    Tramp: float

    def _envelope(self,tr):
        # we cannot pass functions to numba, so make a closure
        if not hasattr(self,'_jit_envelope'):
            ramponfunc = njit(self.ramponfunc)
            ramponfuncpr = njit(self.ramponfuncpr)
            @vectorize(nopython=True)
            def _jit_envelope(tr,E0,Tflat,Tramp,getprime):
                if abs(tr) > Tflat/2 + Tramp:
                    return 0.
                elif abs(tr) > Tflat/2:
                    trel = (Tramp + Tflat/2 - abs(tr)) / Tramp
                    if getprime:
                        return -E0*np.sign(tr)*ramponfuncpr(trel) / Tramp
                    else:
                        return E0*ramponfunc(trel)
                else:
                    return 0. if getprime else E0
            self._jit_envelope = _jit_envelope
        env   = self._jit_envelope(tr,self.E0,self.Tflat,self.Tramp,False)
        envpr = self._jit_envelope(tr,self.E0,self.Tflat,self.Tramp,True)
        return env,envpr

    start_time = property(lambda self: self.t0 - self.Tflat/2 - self.Tramp)
    end_time   = property(lambda self: self.t0 + self.Tflat/2 + self.Tramp)

class LinearFlatTopLaserField(FlatTopLaserField):
    ramponfunc   = staticmethod(lambda trel: trel)
    ramponfuncpr = staticmethod(lambda trel: 1.)
    def _envelope_fourier(self,omega):
        if self.chirp != 0.:
            raise NotImplementedError('Fourier transform of "linear" field with chirp not implemented!')
        return self.E0 * sqrt(8/π) * np.sinc(omega*self.Tramp/(2*π)) * np.sinc(omega*(self.Tramp+self.Tflat)/(2*π)) * (self.Tramp+self.Tflat)/4
    Teff = lambda self,n_photon: self.Tflat + 2*self.Tramp / (1+2*n_photon)

class Linear2FlatTopLaserField(FlatTopLaserField):
    ramponfunc   = staticmethod(lambda trel: sin(π/2*trel)**2)
    ramponfuncpr = staticmethod(lambda trel: sin(π*trel) * π/2)
    def _envelope_fourier(self,omega):
        if self.chirp != 0.:
            raise NotImplementedError('Fourier transform of "linear2" field with chirp not implemented!')
        return self.E0 * sqrt(2*π**3) * cos(omega*self.Tramp/2) * np.sinc(omega*(self.Tramp+self.Tflat)/(2*π)) * (self.Tramp+self.Tflat)/ (2*π**2 - 2*self.Tramp**2*omega**2)
    Teff = lambda self,n_photon: self.Tflat + 2*self.Tramp * gamma(0.5+n_photon*2) / (sqrt(π)*gamma(1+n_photon*2))

@dataclass
class InterpolatingLaserField(LaserField):
    datafile: str

    # this overrides the standard dataclass constructor, so make sure that we set all members
    def __init__(self,is_vecpot,datafile):
        self.is_vecpot = is_vecpot
        self.datafile = datafile

        # print('# Reading laserfield from file:', datafile)
        data = np.loadtxt(datafile,unpack=True)
        if data.shape[0] != 2:
            raise ValueError(f"Laser field datafile '{datafile}' must contain two columns: time and field")
        tt, ff = data

        # print('# Number of data points found:', len(tt))
        if not np.all(tt[1:] >= tt[:-1]):
            raise ValueError("ERROR: times in data file for laser field must be monotonically increasing!")

        if self.is_vecpot:
            self._A = InterpolatedUnivariateSpline(tt,ff,k=4,ext='zeros')
            self._mE = self._A.derivative()
            self._E = lambda t: -self._mE(t)
        else:
            self._E = InterpolatedUnivariateSpline(tt,ff,k=3,ext='zeros')
            self._mA = self._E.antiderivative()
            self._A = lambda t: -self._mA(t)

        # guess parameters
        # find zero-crossings and maxima of the field
        if self.is_vecpot:
            zero_crossings = self._mE.roots()
            Etmp = InterpolatedUnivariateSpline(tt,self._E(tt),k=4,ext='zeros')
        else:
            zero_crossings = self._E.roots()
            Etmp = InterpolatedUnivariateSpline(tt,ff,k=4,ext='zeros')

        tmaxs = Etmp.derivative().roots()
        self._TX = 2*np.diff(zero_crossings).min()
        self.ω0 = 2*π / self.TX
        self.t0 = tmaxs[abs(Etmp(tmaxs)).argmax()]
        self.duration = tt[-1] - tt[0]
        self.E0 = abs(Etmp(self.t0))
        self.chirp = 0.
        self.ϕ0 = 0.

    start_time = property(lambda self: self.tt[0])
    end_time   = property(lambda self: self.tt[-1])
    TX = property(lambda self: self._TX)

    def E(self,t):
        return self._E(t)
    def A(self,t):
        return self._A(t)

def select_param(args, param_names, default=None):
    n = sum(1 for name in param_names if name in args)
    if n > 1:
        raise ValueError(f"Cannot specify more than one out of {', '.join(param_names)}\npassed arguments: {args}.")
    for name, func in param_names.items():
        if name in args:
            return func()
    if default is None:
        raise ValueError(f"You need to specify one out of: {', '.join(param_names)}!")
    return default

def make_laserfield(*, form: str, is_vecpot: bool, **kwargs):
    if form == 'readin':
        return InterpolatingLaserField(is_vecpot,kwargs['datafile'])

    args = dict(is_vecpot=is_vecpot)
    args['E0'] = select_param(kwargs, {'E0': lambda: kwargs['E0'], 'intensity_Wcm2': lambda: np.sqrt(kwargs['intensity_Wcm2'] * au_wcm2toel2)})
    args['ω0'] = select_param(kwargs, {'ω0': lambda: kwargs['ω0'], 'omega': lambda: kwargs['omega'], 'lambda_nm': lambda: 2*np.pi*au_c / (kwargs['lambda_nm'] * au_nm)})
    args['φ0'] = select_param(kwargs, {'φ0': lambda: kwargs['φ0'], 'ϕ0': lambda: kwargs['ϕ0'], 'phase_pi': lambda: np.pi*kwargs['phase_pi']}, 0.)
    args['chirp'] = select_param(kwargs, {'chirp': lambda: kwargs['chirp'], 'linear_chirp_rate_w0as': lambda: args['ω0'] * kwargs['linear_chirp_rate_w0as'] / au_as}, 0.)
    args['t0'] = select_param(kwargs, {'t0': lambda: kwargs['t0'], 'peak_time': lambda: kwargs['peak_time'], 'peak_time_as': lambda: kwargs['peak_time_as'] * au_as})
    duration = select_param(kwargs, {'duration': lambda: kwargs['duration'], 'duration_as': lambda: kwargs['duration_as'] * au_as})
    Tramp = select_param(kwargs, {'Tramp': lambda: kwargs['Tramp'], 'rampon': lambda: kwargs['rampon'], 'rampon_as': lambda: kwargs['rampon_as'] * au_as}, 0.)

    if   form in ('gaussian','gaussianF'):
        # convert from FWHM of field to standard deviation of field
        args["σ"] = duration / sqrt(log(256))
        return GaussianLaserField(**args)
    elif form in ('gaussian2','gaussianI'):
        # convert from FWHM of intensity to standard deviation of field
        args["σ"] = duration / sqrt(log(16))
        return GaussianLaserField(**args)
    elif form in ('sin2','sin4','sin_exp'):
        args["T"] = duration
        args["exponent"] = 2 if form=='sin2' else (4 if form=='sin4' else kwargs['form_exponent'])
        return SinExpLaserField(**args)
    elif form in ('linear','linear2'):
        args["Tflat"] = duration
        args["Tramp"] = Tramp
        lftype = LinearFlatTopLaserField if form=='linear' else Linear2FlatTopLaserField
        return lftype(**args)
    else:
        raise ValueError(f"Unknown laser field form '{form}'")