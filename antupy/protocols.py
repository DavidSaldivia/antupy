from __future__ import annotations
from typing import Protocol, Optional
from antupy.units import Variable

class Models(Protocol):
    def run_simulation(self) -> dict: ...

#-----------------
# utils protocols
class Fluid(Protocol):
    rho: Variable
    cp: Variable
    k: Variable

#----------------
# Subcomponent protocols
class Heater(Protocol):
    nom_power: Variable
    eta: Variable

class Tank(Protocol):
    vol: Variable
    height: Variable
    height_inlet: Variable
    height_outlet: Variable
    height_heater: Variable
    height_thermostat: Variable
    U: Variable
    
    @property
    def diam(self) -> Variable: ...
    @property
    def area_loss(self) -> Variable: ...

class TempControl(Protocol):
    temp_max: Optional[Variable]
    temp_min: Optional[Variable]
    temp_deadband: Optional[Variable]
    temp_consump: Optional[Variable]

#----------------
# Component protocols
class HotWaterHeater(Protocol):
    #metadata
    name: str
    model: str
    cost: Variable

    #subcomponents
    heater: Heater
    tank: Optional[Tank]
    control: TempControl
    fluid: Fluid

    #properties
    @property
    def thermal_cap(self) -> Variable: ...
    
    @classmethod
    def from_model_file(cls) -> HotWaterHeater: ...

#----------------
# System protocols

