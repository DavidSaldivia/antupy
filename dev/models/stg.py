from dataclasses import dataclass

import numpy as np
import pandas as pd

from antupy.protocols import Model
from antupy.utils.props import Water
from antupy import (
    Var,
    Array,
    CF
)

@dataclass
class ThermalStorageTank():
    name = "Cilyndrical Thermal Storage Tank"
    geometry = "cylinder"
    fluid = Water()
    model = "0D"

    # tank geometry and losses
    vol = Var(0.315,"m3")
    height = Var(1.45, "m")
    height_inlets = Array([0.113], "m")
    height_outlets = Array([1.317], "m")
    height_heaters = Array([0.103], "m")
    height_thermostats = Array([0.103], "m")
    U = Var(0.9, "W/m2-K")

    #numerical simulation
    nodes = 10
    temps_ini = Array(45.0*np.ones(nodes), "degC")

    # control
    temp_max = Var(65.0, "degC")
    temp_deadband = Var(10.0, "degC")
    temp_min = Var(45.0, "degC")
    temp_consump = Var(45.0, "degC")

    def __post_init__(self):
        temps = self.temps_ini

    @property
    def thermal_cap(self) -> Var:
        rho = self.fluid.rho()
        cp = self.fluid.cp()
        thermal_cap = self.vol * (rho * cp) * (self.temp_max - self.temp_min)
        return thermal_cap.su("kWh")

    @property
    def diam(self) -> Var:
        vol = self.vol
        height = self.height
        diam = (4 * self.vol / np.pi / self.height).gv("m2") ** 0.5
        return Var( diam , "m" )

    @property
    def area_loss(self) -> Var:
        diam = self.diam
        height = self.height
        area_loss = np.pi * diam * (diam / 2 + height)
        return area_loss.su("m2")

    def run_model(self, ts: pd.DataFrame):
        match self.model:
            case "0D":
                df_tm = _storage_tank_0D(self, ts)
            case "1D":
                df_tm = _storage_tank_1D(self, ts)
            case _:
                raise KeyError(f"specificied model ({self.model}) is not a valid one")


def _storage_tank_0D(tank: ThermalStorageTank, data: pd.DataFrame):

    cp = tank.fluid.cp().get_value("J/kg-K")
    rho = tank.fluid.rho().get_value("kg/m3")
    k = tank.fluid.k().get_value("W/m-K")
    vol = tank.vol.get_value("m3")
    U = tank.U.get_value("W/m2-K")
    area_loss = tank.area_loss.get_value("m2")
    temp_tank = np.mean(tank.temps_ini.gv("degC"))
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

