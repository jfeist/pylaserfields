"""Python library for describing time-dependent laserfields by Johannes Feist."""

__version__ = "0.4.2"

__all__ = [
    "intensity_Wcm2_to_Eau",
    "make_laserfield",
    "GaussianLaserField",
    "SinExpLaserField",
    "LinearFlatTopLaserField",
    "Linear2FlatTopLaserField",
    "InterpolatingLaserField",
    "LaserFieldCollection",
]

from .laserfields import (
    intensity_Wcm2_to_Eau,
    make_laserfield,
    GaussianLaserField,
    SinExpLaserField,
    LinearFlatTopLaserField,
    Linear2FlatTopLaserField,
    InterpolatingLaserField,
    LaserFieldCollection,
    au_as as au_as,
    au_wcm2toel2 as au_wcm2toel2,
    au_wcm2 as au_wcm2,
    au_m as au_m,
    au_cm as au_cm,
    au_nm as au_nm,
    au_c as au_c,
    au_eV as au_eV,
    au_m_He as au_m_He,
    au_m_n as au_m_n,
)
