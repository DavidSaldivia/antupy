from dataclasses import dataclass

import numpy as np
import pandas as pd

from antupy.protocols import Model
from antupy.props import ConstantWater
from antupy.units import (
    Variable,
    Array,
    conversion_factor as CF
)

@dataclass
class ThermalStorageTank():
    name = "Cilyndrical Thermal Storage Tank"
    geometry = "cylinder"
    fluid = ConstantWater()
    model = "0D"

    # tank geometry and losses
    vol = Variable(0.315,"m3")
    height = Variable(1.45, "m")
    height_inlets = Array([0.113], "m")
    height_outlets = Array([1.317], "m")
    height_heaters = Array([0.103], "m")
    height_thermostats = Array([0.103], "m")
    U = Variable(0.9, "W/m2-K")

    #numerical simulation
    nodes = 10
    temps_ini = Array(45.0*np.ones(nodes), "degC")

    # control
    temp_max = Variable(65.0, "degC")
    temp_deadband = Variable(10.0, "degC")
    temp_min = Variable(45.0, "degC")
    temp_consump = Variable(45.0, "degC")

    def __post_init__(self):
        temps = self.temps_ini

    @property
    def thermal_cap(self) -> Variable:
        vol = self.vol.get_value("m3")
        rho = self.fluid.rho.get_value("kg/m3")
        cp = self.fluid.cp.get_value("J/kg-K")
        temp_max = self.temp_max.get_value("degC")
        temp_min = self.temp_min.get_value("degC")
        thermal_cap = vol * (rho * cp) * (temp_max - temp_min) * CF("J","kWh")
        return Variable( thermal_cap, "kWh")

    @property
    def diam(self) -> Variable:
        vol = self.vol.get_value("m3")
        height = self.height.get_value("m")
        diam = (4 * vol / np.pi / height) ** 0.5
        return Variable( diam , "m" )
    
    @property
    def area_loss(self) -> Variable:
        diam = self.diam.get_value("m")
        height = self.height.get_value("m")
        area_loss = np.pi * diam * (diam / 2 + height)
        return Variable( area_loss, "m2" ) 

    def run_model(self, ts: pd.DataFrame):
        match self.model:
            case "0D":
                df_tm = _storage_tank_0D(self, ts)
            case "1D":
                df_tm = _storage_tank_1D(self, ts)
            case _:
                raise KeyError(f"specificied model ({self.model}) is not a valid one")


def _storage_tank_0D(tank: ThermalStorageTank, data: pd.DataFrame):

    cp = tank.fluid.cp.get_value("J/kg-K")
    rho = tank.fluid.rho.get_value("kg/m3")
    k = tank.fluid.k.get_value("W/m-K")
    vol = tank.vol.get_value("m3")
    U = tank.U.get_value("W/m2-K")
    area_loss = tank.area_loss.get_value("m2")
    temp_tank = np.mean(tank.temps.get_values("degC"))
    nodes = tank.nodes

    dt = data["dt"]
    m_in = data["m_in"]
    temp_in = data["temp_in"]
    m_out = data["m_out"]
    temp_out = data["temp_out"]
    temp_amb = data["temp_amb"]

    heat_loss = U * area_loss * (temp_tank - temp_amb)
    
    heat_in = m_in * (cp*temp_in - cp*temp_tank) 
    heat_out = m_out * (cp*temp_tank - cp*temp_out) 
    
    temp_new = temp_tank + ( dt / (vol * rho * cp) ) * ( heat_in - heat_out - heat_loss )

    return temp_new

def _storage_tank_1D(tank: ThermalStorageTank, ts: pd.DataFrame):
    pass


class Battery():
    def __init__(self):
        self.name = "Battery (Electrochemical storage)"


class CAES():
    def __init__(self):
        self.name = "Compressed Air Storage"


class ChemicalStorage():
    def __init__(self):
        self.name = "Chemical Storage"

