from antupy.core import Var, CF

def test_CF():
    assert CF("m", "km").v == 0.001
    assert CF("W", "kJ/hr").v == 3.6
    assert CF("m3/s", "L/min").v == 60000.
    assert CF("m", "km") == Var(0.001, "km/m")
    assert CF("W", "kJ/hr") == Var(3.6, "kJ/hr-W")
    assert CF("m3/s", "L/min") == Var(60000., "L-s/min-m3")


def test_conversion_time():
    time_sim = Var(365, "day")
    assert time_sim.u('day') == 365
    assert time_sim.u('hr') == 8760.
    assert time_sim.u('s') == 31536000.0


def test_conversion_temp():
    temp1 = Var(300., "K")
    temp2 = Var(20., "°C")
    assert temp1.u("°C") == 300-273.15
    assert temp1.u("K") == 300.
    assert temp2.u("°C") == 20.
    assert temp2.u("K") == 20+273.15


def test_add_vars():
    time_sim1 = Var(365,"day")
    time_sim2 = Var(24, "hr")
    assert (time_sim1 + time_sim2).u("day") == 366


def test_mul_vars():
    time_sim = Var(1, "day")
    nom_power = Var(100, "kW")
    energy = nom_power * time_sim
    assert energy.u("kW-hr") == 24*100
    assert energy.u("kWh") == 24*100
    assert energy.u("kJ") == 24*100*3600
    assert (8*energy/2 - energy*2) == 2*Var(100*24., "kWh")
