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
from .laserfields import au_as as au_as
from .laserfields import au_c as au_c
from .laserfields import au_cm as au_cm
from .laserfields import au_eV as au_eV
from .laserfields import au_m as au_m
from .laserfields import au_m_He as au_m_He
from .laserfields import au_m_n as au_m_n
from .laserfields import au_nm as au_nm
from .laserfields import au_wcm2 as au_wcm2
from .laserfields import au_wcm2toel2 as au_wcm2toel2
