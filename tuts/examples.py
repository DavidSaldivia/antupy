import antupy as ap
import numpy as np

from antupy.core import units
from antupy.core.units import Unit


def units_available():

    print("BASE UNITS:")
    for v,k in units.BASE_UNITS.items():
        print(v,k)

    print("DERIVED UNITS:")
    for v,k in units.DERIVED_UNITS.items():
        print(v,k)

    print("RELATED UNITS:")
    for v,k in units.RELATED_UNITS.items():
        print(v,k)

    print("SI PREFIXES:")
    for v,k in units.PREFIXES.items():
        print(v,k)

    return None


def valid_unit_strings():
    from antupy import Unit
    # Valid unit strings
    Unit("m")
    Unit("m-s")
    Unit("km2")
    Unit("kJ")
    Unit("kg-m/s2")
    Unit("TW-hr")

    # Invalid unit strings (throws an error)
    # Unit("m s")
    # Unit("m*s")
    # Unit("m/s2/s3")
    # Unit("(m/s)")
    # Unit("m-s-2")

def base_representation():
    u = Unit("W/kg-K")
    print(u.base_factor)
    print(u.base_units)
    print(u.si)


def using_var():
    mass = ap.Var(5.0, "kg")
    pressure = ap.Var(101325, "Pa")

    # Using the gv method
    print(mass.get_value("g"))  # Outputs: 5000.0 '[g]'
    print(mass.gv("ton"))  # Outputs: 0.005 '[ton]'
    # print(mass.gv("s"))   # Outputs: ValueError

    # You can perform most of arithmetic operations: Addition and subtraction.
    result = mass + ap.Var(500, "g")  # Adds 0.5 kg to 5 kg
    print(result)  # Outputs: 5.5 '[kg]'

    # Multiplication and division.
    time_sim = ap.Var(1, "day")
    nom_power = ap.Var(100, "kW")
    energy = nom_power * time_sim
    print(energy) # Outputs: 100 [kW-day]
    print(energy.gv("kW-hr")) # Outputs:  2400.0
    print(energy.gv("kWh")) # Outputs: 2400.0
    print(energy.gv("kJ")) # Outputs 8640000.0

    # Using the su method
    print((8*energy/2 - energy*2)) # Outputs: 200 ["kW-day"]
    print((8*energy/2 - energy*2).su("kWh")) # Outputs: 4800 ["kWh"]
    # print((8*energy/2 - energy*2).su("MW"))  # Outputs: Traceback (most recent call last): ...


def using_array():

    # Array is used in a similar way, but with lists.
    # As a general rule, whatever that can go in np.array, can go in ap.Array.
    receiver_powers = ap.Array([10., 20., 40.], "MW")
    receiver_areas = ap.Array(np.linspace(20,50,3), "m2")
    receiver_flux_avg = receiver_powers / receiver_areas
    print(receiver_flux_avg) # Outputs: [0.5        0.57142857 0.8       ] [MW/m2]

    # gv() and su() work similarly to Var, but element-wise.
    print(receiver_areas.su("ha"))
    print(receiver_areas.su("mm2"))

    #Combining Array with Var
    eta_rcv = ap.Var(0.9, "")
    receiver_flux_output = receiver_flux_avg * eta_rcv

    # Array also have some methods for element-wise operations, such as sum, mean, etc.
    print(receiver_flux_avg.mean()) # Outputs a Var


def using_props():

    # Create fluid instance
    water = ap.props.Water()

    # Get properties at saturation conditions
    T = ap.Var(373.15, "K")
    P = ap.Var(200, "kPa")

    # Thermophysical properties
    density = water.rho(T, P)
    specific_heat = water.cp(T, P)
    viscosity = water.viscosity(T, P)
    thermal_conductivity = water.k(T, P)

    # Dimensionless numbers
    prandtl = (specific_heat * viscosity / thermal_conductivity).su("-")

    print(f"Density: {density:.1f}")
    print(f"Enthalpy: {specific_heat:.1f}")
    print(f"Prandtl number: {prandtl:.3f}")


def using_props_2():

    fluid = ap.props.Water()

    temp_max = ap.Var(60, "degC")
    temp_in = ap.Var(20, "degC")
    vol_tank = ap.Var(300, "L")

    temp_avg = (temp_max + temp_in) / 2
    cp  = fluid.cp(temp_avg)
    rho = fluid.rho(temp_avg)

    q_stg = vol_tank * rho * cp * (temp_max - temp_in)
    q_stg = q_stg.su("kWh")
    print(f"Energy stored: {q_stg:.1f}")



def main():
    units_available()
    valid_unit_strings()
    base_representation()
    using_var()
    using_array()
    using_props()
    using_props_2()

if __name__ == "__main__":
    main()