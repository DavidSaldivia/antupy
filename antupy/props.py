from dataclasses import dataclass
from antupy.units import Variable
from typing import Protocol

class Fluid(Protocol):
    rho: Variable
    cp: Variable
    k: Variable


#-------------------------
@dataclass
class ConstantWater(Fluid):
    rho = Variable(1000., "kg/m3")  # density (water)
    cp = Variable(4180., "J/kg-K")  # specific heat (water)
    k = Variable(0.6, "W/m-K")  # thermal conductivity (water)
    def __repr__(self) -> str:
        return "Liquid incompressible water"


@dataclass
class SolarSalt(Fluid):
    rho = Variable(1900., "kg/m3")  # density (water)
    cp = Variable(1100., "J/kg-K")  # specific heat (water)
    k = Variable(0.55, "W/m-K")  # thermal conductivity (water)
    def __repr__(self) -> str:
        return "Liquid incompressible water"