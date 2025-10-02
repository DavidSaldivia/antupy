import math
from antupy.var import Var, CF

def test_CF():
    assert CF("m", "km").v == 0.001
    assert CF("W", "kJ/hr").v == 3.6
    assert CF("m3/s", "L/min").v == 60000.
    assert CF("m", "km") == Var(0.001, "km/m")
    assert CF("W", "kJ/hr") == Var(3.6, "kJ/hr-W")
    assert CF("m3/s", "L/min") == Var(60000., "L-s/min-m3")


def test_conversion_time():
    time_sim = Var(365, "day")
    assert time_sim.gv('day') == 365
    assert time_sim.gv('hr') == 8760.
    assert time_sim.gv('s') == 31536000.0


def test_conversion_temp():
    temp1 = Var(300., "K")
    temp2 = Var(20., "°C")
    assert temp1.gv("°C") == 300-273.15
    assert temp1.gv("K") == 300.
    assert temp2.gv("°C") == 20.
    assert temp2.gv("K") == 20+273.15


def test_add_sub_vars():
    time_sim1 = Var(365,"day")
    time_sim2 = Var(24, "hr")
    assert (time_sim1 + time_sim2).gv("day") == 366
    assert (time_sim1 - time_sim2).gv("day") == 364


def test_mul_div_vars():
    time_sim = Var(1, "day")
    nom_power = Var(100, "kW")
    energy = nom_power * time_sim
    assert energy.gv("kW-hr") == 24*100
    assert energy.gv("kWh") == 24*100
    assert energy.gv("kJ") == 24*100*3600
    assert (8*energy/2 - energy*2) == 2*Var(100*24., "kWh")


def test_comparison_vars():
    assert Var(1000, "m") == Var(1, "km")
    assert Var(100, "m") < Var(1, "km")
    assert Var(1000, "m") <= Var(1, "km") or Var(100, "m") <= Var(1, "km")
    assert Var(1, "day") > Var(1, "hr")
    assert Var(1, "day") >= Var(1, "hr") or Var(1,"day") >= Var(24, "hr")


def test_unary_vars():
    assert -Var(1,"m") == Var(-1, "m")
    assert +Var(1,"m") == Var(1, "m")
    assert abs(Var(-1,"m")) == Var(1, "m")


def test_math_methods():
    assert round(Var(1.141, "m"), 1) == Var(1.1, "m")
    assert round(Var(1.141, "m")) == Var(1, "m")
    assert math.trunc(Var(1.141, "m")) == Var(1, "m")
    assert math.floor(Var(1.141, "m")) == Var(1, "m")
    assert math.ceil(Var(1.141, "m")) == Var(2, "m")

def test_adim_influence():
    power_0 = Var(10, "MW")
    eta_1 = Var(0.9, "-")
    eta_2 = Var(0.8, "-")
    power_1 = power_0 * eta_1
    power_2 = power_0 / eta_2

    assert Var(0.9/0.8, "-") == eta_1 / eta_2
    assert Var(1,"-") == Var(1, "K/K")
    assert power_1 == Var(9, "MW")
    assert power_1.gv("kW") == 9000
    assert power_2 == Var(12.5, "MW")
    assert power_2.gv("kW") == 12500
    assert (power_1 / power_2).su("") == Var(0.72, "-")


def test_var_formatting():
    """Test the __format__ method for f-strings and format() calls."""
    # Basic formatting
    var = Var(3.14159, "m")
    assert f"{var}" == "3.14159 [m]"
    assert f"{var:.2f}" == "3.14 [m]"
    assert f"{var:.3e}" == "3.142e+00 [m]"
    
    # None value formatting
    var_none = Var(None, "kg")
    assert f"{var_none}" == "None [kg]"
    assert f"{var_none:.2f}" == "None [kg]"  # Format spec ignored for None
    
    # Zero value formatting
    var_zero = Var(0, "W")
    assert f"{var_zero:.1f}" == "0.0 [W]"
    
    # Large number formatting
    var_large = Var(1234567.89, "J")
    assert f"{var_large:.2e}" == "1.23e+06 [J]"
    
    # Temperature formatting
    var_temp = Var(25.5, "°C")
    assert f"{var_temp:.1f}" == "25.5 [°C]"
    
    # Width formatting (applies to entire string)
    var_small = Var(3.1, "m")
    formatted = f"{var_small:>15}"
    assert "3.1 [m]" in formatted
    assert len(formatted) == 15