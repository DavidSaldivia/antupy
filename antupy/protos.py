from __future__ import annotations
import pandas as pd
from typing import Protocol, Self

# main classes
class Analyser(Protocol):
    def input(self):
        ...
    def output(self) -> dict[str,float|dict]:
        ...


class Model(Protocol):
    solver: Solver
    @classmethod
    def new_instance(cls, *args) -> Model:
        ...
    def run_simulation(self, ts:pd.DataFrame) -> pd.DataFrame:
        ...


class TimeSeriesGenerator(Protocol):
    @classmethod
    def parameters(cls, dict) -> Self:
        ...
    def load_data(self, cols:list[str]) -> pd.DataFrame:
        ...


class Solver(Protocol):
    def run_simulation(self, model: Model, ts: pd.DataFrame) -> pd.DataFrame:
        ...


# # utils protocols
# class Fluid():
#     rho: Variable
#     cp: Variable
#     k: Variable

# #----------------
# # Subcomponent protocols
# class Heater():
#     nom_power: Variable
#     eta: Variable

# class Tank():
#     vol: Variable
#     height: Variable
#     height_inlet: Variable
#     height_outlet: Variable
#     height_heater: Variable
#     height_thermostat: Variable
#     U: Variable
    
#     @property
#     def diam(self) -> Variable: ...
#     @property
#     def area_loss(self) -> Variable: ...

# class TempControl(Protocol):
#     temp_max: Optional[Variable]
#     temp_min: Optional[Variable]
#     temp_deadband: Optional[Variable]
#     temp_consump: Optional[Variable]

# #----------------
# # Component protocols
# class HotWaterHeater(Protocol):
#     #metadata
#     name: str
#     model: str
#     cost: Variable

#     #subcomponents
#     heater: Heater
#     tank: Optional[Tank]
#     control: TempControl
#     fluid: Fluid

#     #properties
#     @property
#     def thermal_cap(self) -> Variable:
#         ...
    
#     @classmethod
#     def from_model_file(cls) -> HotWaterHeater:
#         ...
