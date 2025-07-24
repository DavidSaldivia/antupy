import numpy as np
from antupy import Var
from antupy.props import Air
from antupy.props import DryAir
from antupy.props import HumidAir
from antupy.props import SaturatedWater
from antupy.props import SaturatedSteam
from antupy.props import SeaWater

def test_prop_dry_air():
    air = DryAir()
    temp = Var(300., "K")
    assert air.rho(temp) == air.rho(temp)
    assert air.cp(temp) == air.cp(temp)
    assert air.k(temp) == air.k(temp)
    assert np.round(air.rho(temp).gv("kg/m3"), 3) == 1.176
    assert np.round(air.cp(temp).gv("kJ/kg-K"), 3) == 1.005
    assert np.round(air.k(temp).gv("W/m-K"), 3) == 0.026
    assert np.round(air.viscosity(temp).gv("m2/s") * 1e5, 3) == 1.573

def test_prop_air():
    air = Air()
    temp = Var(300., "K")
    pressure = Var(1, "atm")
    assert air.rho(temp, pressure) == air.rho(temp, pressure)
    assert air.cp(temp, pressure) == air.cp(temp, pressure)
    assert air.k(temp, pressure) == air.k(temp, pressure)
    assert np.round(air.rho(temp, pressure).gv("kg/m3"), 3) == 1.177
    assert np.round(air.cp(temp, pressure).gv("kJ/kg-K"), 3) == 1.006
    assert np.round(air.k(temp, pressure).gv("W/m-K"), 3) == 0.026
    assert np.round(air.viscosity(temp, pressure).gv("Pa-s") * 1e5, 3) == 1.854


def test_prop_humid_air():
    air = HumidAir()
    temp = Var(300., "K")
    pressure = Var(101325, "Pa")
    abshum = Var(0.01, "-")
    assert air.rho(temp, pressure, abshum) == air.rho(temp, pressure, abshum)
    assert air.cp(temp, abshum) == air.cp(temp, abshum)
    assert air.k(temp, abshum) == air.k(temp, abshum)
    assert np.round(air.rho(temp, pressure, abshum).gv("kg/m3"), 3) == 1.146
    assert np.round(air.cp(temp, abshum).gv("kJ/kg-K"), 3) == 1.024
    assert np.round(air.k(temp, abshum).gv("W/m-K"), 3) == 0.026

    # assert np.round(air.viscosity(temp,abshum).u("Pa-s") * 1e5, 3) == 1.573
    # viscosity is not implemented correctly. Check the equation in props.py


def test_prop_sat_water():
    water = SaturatedWater()
    temp = Var(300., "K")
    assert water.rho(temp) == water.rho(temp)
    assert water.cp(temp) == water.cp(temp)
    assert water.k(temp) == water.k(temp)
    assert water.k(temp)/(water.rho(temp)*water.cp(temp)) == water.k(temp)/(water.rho(temp)*water.cp(temp))
    assert np.round(water.rho(temp).gv("kg/m3"), 1) == 996.4
    assert np.round(water.cp(temp).gv("kJ/kg-K"), 2) == 4.18
    assert np.round(water.k(temp).gv("W/m-K"), 3) == 0.61
    # assert np.round(water.viscosity(temp).u("Pa-s") * 1e5, 3) == 1.573

def test_prop_sat_steam():
    steam = SaturatedSteam()
    temp = Var(99.63, "°C")
    assert np.round(steam.rho(temp).gv("kg/m3"), 3) == 0.591
    assert np.round(steam.cp(temp).gv("kJ/kg-K"), 2) == 2.08
    assert np.round(steam.k(temp).gv("W/m-K"), 3) == 0.025

def test_prop_seawater():
    seawater = SeaWater()


if __name__ == "__main__":
    test_prop_dry_air()
    test_prop_humid_air()
    test_prop_sat_water()
    test_prop_sat_steam()
    test_prop_seawater()