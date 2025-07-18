from antupy.units import Unit
from antupy.core import Var, CF
from antupy.units import BASE_UNITS, DERIVED_UNITS, RELATED_UNITS, PREFIXES

def test_unit_uniqueness():
    UNITS_NOPREFIX = list((BASE_UNITS|DERIVED_UNITS|RELATED_UNITS).keys())
    UNITS = [ f"{prefix}{unit}" for prefix in PREFIXES for unit in UNITS_NOPREFIX ]
    assert len(UNITS) == len(set(UNITS)), "Units are not unique"

def test_unit_split_comps():
    unit_label = "kJ/kg-K"
    factor, unit_list = Unit._split_unit(unit_label)
    assert factor == 1e0 and unit_list == [("J", 1), ("g", -1), ("K", -1)]

def test_unit_translation():
    for (a,b) in [
        (Unit("kg-m2/s2"), Unit("m2-kg/s2")),
        (Unit("kg/m-K"), Unit("g/mm-K")),
        (Unit("W"), Unit("J/s")),
        (Unit("Hz"), Unit("1/s")),
        (Unit("kW-ks"), Unit("MJ")),
        (Unit("J/kg-K"), Unit("m2/s2-K", 1e0)),
        (Unit("g/mol"), Unit("kg/kmol")),
        (Unit("g/mol"), Unit("kg/mol",1e-3)),
        (Unit("min"), Unit("s", 60.)),
        (Unit("hr"), Unit("s", 3600.)),
        (Unit("yr"), Unit("s", 3600.*365*24)),
        (Unit("kJ"), Unit("kg-m2/s2",1e3)),
        (Unit("m"), Unit("m3/m2")),
        (Unit("1/m"), Unit("m2/m3")),
        (Unit("s"), Unit("sec")),
        (Unit("1/s"), Unit("1/sec")),
        (Unit("m/s"), Unit("km/hr", 3.6)),
        (Unit("1/hr"), Unit("1/s", 1/3600.)),
        (Unit("Wh"), Unit("J", 3600)),
        (Unit("Wh",1/3600), Unit("J")),
        (Unit("cal"), Unit("J",4184.)),
        (Unit("1/cal"), Unit("1/J",1/4184.)),
        (Unit("ha"), Unit("m2",1e4)),
        (Unit("ha"), Unit("m-m",1e4)),
        (Unit("degC"), Unit("K")),
        (Unit("g/L"), Unit("kg/m3")),
        (Unit("lm"), Unit("cd-sr")),
        # (Unit("km2"), Unit("m2", 1e6)),
    ]:
        # print(f"{a}({a.si}),{b}({b.si})")
        assert a==b, f"{a},{b}"

def test_CF():
    assert CF("m", "km").v == 0.001
    assert CF("W", "kJ/hr").v == 3.6
    assert CF("m3/s", "L/min").v == 60000.
    assert CF("m", "km") == Var(0.001, "km/m")
    assert CF("W", "kJ/hr") == Var(3.6, "kJ/hr-W")
    assert CF("m3/s", "L/min") == Var(60000., "L-s/min-m3")

def test_conversion_temp():
    temp1 = Var(300., "K")
    temp2 = Var(20., "°C")
    assert temp1.u("°C") == 300-273.15
    assert temp1.u("K") == 300.
    assert temp2.u("°C") == 20.
    assert temp2.u("K") == 20+273.15

def test_conversion_time():
    time_sim = Var(365, "day")
    assert time_sim.u('day') == 365
    assert time_sim.u('hr') == 8760.
    assert time_sim.u('s') == 31536000.0

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


def main():
    test_unit_translation()
    test_mul_vars()

if __name__ == "__main__":
    main()