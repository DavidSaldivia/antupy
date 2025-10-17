from antupy import Unit
from antupy.core.units import BASE_UNITS, DERIVED_UNITS, RELATED_UNITS, PREFIXES

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
        (Unit("km2"), Unit("m2", 1e6)),
        (Unit("ha"), Unit("m2", 1e4)),
        (Unit("mm2"), Unit("m2", 1e-6)),
    ]:
        # print(f"{a}({a.si}),{b}({b.si})")
        assert a==b, f"{a},{b}"

def main():
    test_unit_translation()

if __name__ == "__main__":
    main()