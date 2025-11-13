import numpy as np
from antupy import Var
from antupy.utils.props import Air
from antupy.utils.props import DryAir
from antupy.utils.props import HumidAir
from antupy.utils.props import SaturatedWater
from antupy.utils.props import SaturatedSteam
from antupy.utils.props import SeaWater

def test_prop_dry_air():
    air = DryAir()
    temp = Var(300., "K")
    assert air.rho(temp) == air.rho(temp)
    assert air.cp(temp) == air.cp(temp)
    assert air.k(temp) == air.k(temp)
    assert round(air.rho(temp),3) == Var(1.176, "kg/m3")
    assert round(air.cp(temp), 3) == Var(1.005, "kJ/kg-K")
    assert round(air.k(temp), 3) == Var(0.026, "W/m-K")
    assert np.round(air.viscosity(temp).gv() * 1e5, 3) == 1.573

def test_prop_air():
    air = Air()
    temp = Var(300., "K")
    pressure = Var(1, "atm")
    assert air.rho(temp, pressure) == air.rho(temp, pressure)
    assert air.cp(temp, pressure) == air.cp(temp, pressure)
    assert air.k(temp, pressure) == air.k(temp, pressure)
    assert round(air.rho(temp, pressure), 3) == Var(1.177, "kg/m3")
    assert round(air.cp(temp, pressure), 0) == Var(1006, "J/kg-K")
    assert round(air.k(temp, pressure), 3) == Var(0.026, "W/m-K")
    assert np.round(air.viscosity(temp, pressure).gv("Pa-s") * 1e5, 3) == 1.854

def test_prop_humid_air():
    air = HumidAir()
    temp = Var(300., "K")
    pressure = Var(101325, "Pa")
    abshum = Var(0.01, "-")
    assert air.rho(temp, pressure, abshum) == air.rho(temp, pressure, abshum)
    assert air.cp(temp, abshum) == air.cp(temp, abshum)
    assert air.k(temp, abshum) == air.k(temp, abshum)
    assert round(air.rho(temp, pressure, abshum), 3) == Var(1.146, "kg/m3")
    assert round(air.cp(temp, abshum), 3) == Var(1024, "J/kg-K")
    assert round(air.k(temp, abshum), 3) == Var(0.026, "W/m-K")

    # assert np.round(air.viscosity(temp,abshum).u("Pa-s") * 1e5, 3) == 1.573
    # viscosity is not implemented correctly. Check the equation in props.py


def test_prop_sat_water():
    water = SaturatedWater()
    temp = Var(300., "K")
    assert water.rho(temp) == water.rho(temp)
    assert water.cp(temp) == water.cp(temp)
    assert water.k(temp) == water.k(temp)
    assert (
        water.k(temp)/(water.rho(temp)*water.cp(temp)) 
        == water.k(temp)/(water.rho(temp)*water.cp(temp))
    )
    assert round(water.rho(temp), 1) == Var(996.4, "kg/m3")
    assert round(water.cp(temp), 2) == Var(4180.11, "J/kg-K")
    assert round(water.k(temp), 3) == Var(0.61, "W/m-K")
    # assert np.round(water.viscosity(temp).u("Pa-s") * 1e5, 3) == 1.573

def test_prop_sat_steam():
    steam = SaturatedSteam()
    temp = Var(99.63, "°C")
    assert round(steam.rho(temp), 3) == Var(0.591, "kg/m3")
    assert round(steam.cp(temp), 2) == Var(2078.55, "J/kg-K")
    assert round(steam.k(temp), 3) == Var(0.025, "W/m-K")

def test_prop_seawater():
    seawater = SeaWater()
    temp = Var(100., "degC")
    assert round(seawater.rho(temp), 3) == Var(984.29, "kg/m3")
    assert round(seawater.cp(temp), 2) == Var(4043.63, "J/kg-K")
    # assert round(seawater.k(temp), 3) == Var(0.61, "W/m-K")


if __name__ == "__main__":
    test_prop_dry_air()
    test_prop_humid_air()
    test_prop_sat_water()
    test_prop_sat_steam()
    test_prop_seawater()