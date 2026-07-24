"""Python library for describing time-dependent laserfields by Johannes Feist."""

__version__ = "0.5.0"

__all__ = [
    "GaussianLaserField",
    "InterpolatingLaserField",
    "LaserFieldCollection",
    "Linear2FlatTopLaserField",
    "LinearFlatTopLaserField",
    "SinExpLaserField",
    "intensity_Wcm2_to_Eau",
    "make_laserfield",
]

from .laserfields import (
    GaussianLaserField,
    InterpolatingLaserField,
    LaserFieldCollection,
    Linear2FlatTopLaserField,
    LinearFlatTopLaserField,
    SinExpLaserField,
    intensity_Wcm2_to_Eau,
    make_laserfield,
)
from .laserfields import (
    au_as as au_as,
    au_c as au_c,
    au_cm as au_cm,
    au_eV as au_eV,
    au_m as au_m,
    au_m_He as au_m_He,
    au_m_n as au_m_n,
    au_nm as au_nm,
    au_wcm2 as au_wcm2,
    au_wcm2toel2 as au_wcm2toel2,
)
