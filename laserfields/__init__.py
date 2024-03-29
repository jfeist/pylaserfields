"""Python library for describing time-dependent laserfields by Johannes Feist."""

__version__ = '0.3.1'

__all__ = ['intensity_Wcm2_to_Eau','make_laserfield', 'GaussianLaserField', 'SinExpLaserField',
           'LinearFlatTopLaserField', 'Linear2FlatTopLaserField', 'InterpolatingLaserField']

from .laserfields import (intensity_Wcm2_to_Eau, make_laserfield, GaussianLaserField, SinExpLaserField,
                          LinearFlatTopLaserField, Linear2FlatTopLaserField, InterpolatingLaserField,
                          au_as, au_wcm2toel2, au_wcm2, au_m, au_cm, au_nm, au_c, au_eV, au_m_He, au_m_n)
