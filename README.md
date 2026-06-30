# antupy
`antupy` (from the *mapuzugun* word "antü" (sun)[^1]) is an open-source python toolkit to support simulations of energy engineering projects. It is structured in three layers:
 - **Core layer**: antupy works with a unit management system (`Unit`) based on the SI-unit system that allows to handle physical quantities. Based on that, three classes are available: `Var` for scalars, `Array` for 1D vectors/timeseries, and `Frame` for polars(pandas) tabular data with per-column units.
 - **Utilities layer**: A set of helper modules: `props` (thermophysical properties), `htc` (heat transfer correlations), `solar` (sun position and radiation calculations), and `loc` (geographical location management).
 - **Simulation layer**: Framework for building and analyzing energy systems using `Model`, `Plant`, and `Parametric` classes, and a set of time-series generators to retrieve useful data (e.g. `Weather`, `Market`).


## Documentation
The full documentation is available [here](https://antupy.readthedocs.io/).

## Brief Introduction
`antupy` works in its core with a unit management module `units`, which include the class `Unit` to represent units that are compatible with the SI unit system. From this, three type of variables are introduced:
 1. The `Var` class to manage single variables, with the structure `(value:float, unit:str)`.
 2. The `Array` class for 1D data structures in the form of `(array:np.ndarray, unit:str)`.
 3. The `Frame` class for 2D data structures in the form of `(frame:pd.DataFrame, units:list[str])`.
Where the string `unit` (or `units`) has to follow a couple of [simple rules](https://antupy.readthedocs.io/en/latest/units.html#valid-unit-strings) to represent properly physical units. All three classes support arithmetic operations with automatic unit conversion and dimensional checking, ensuring dimensional consistency throughout calculations.

## Examples

### Quick Start - Core Classes

```python
import antupy as ap
import numpy as np

# Scalar with units
mass = ap.Var(5.0, "kg")
power = ap.Var(100, "kW")
time = ap.Var(1, "day")
eta =  ap.Var(0.8, "-")

# Arrays and Frames with units
temps = ap.Array([20, 25, 30], "degC")

# Frames with per-column units
data = {"power": [10., 20., 40.], "area": [20., 35., 50.]}
df = ap.Frame(data=data, units={"power": "MW", "area": "m2"})

```
All the core classes have two main methods to interact with them: `.gv(str)` (or `.get_value(str)`, where `str` is any valid unit string) and `.su(str)` (or `.set_unit(str)`). `.gv(str)` allows you to retrieve the data (as float, np.ndarray, or pl.DataFrame) from your antupy variables, while `.su(str)` allows you to change the units in which the data is stored. This is useful to check wheter a variable has the units you expect. You can also use the `compatible()` method, if you are not sure the unit of a variable. You can also retrieve a variable label string using the `.u` (or `.unit`) attribute and the stored data with the `.v` ( or `.value`) attribute. The difference between `.gv()` and `.v` is the last one does not check the units, so use it carefully and under your own responsability.

```python
import antupy as ap

# Scalar with units
mass = ap.Var(5.0, "kg")
power = ap.Var(100, "kW")
time = ap.Var(1, "day")
temps = ap.Array([20, 25, 30], "degC")

mass_in_ton = mass.gv("ton")
power_wrong = power.gv("J")         # Throws and error. "J" not compatible with "kW".


mass2 = mass.su("mg")
energy = (power * time).su("J")     # The expected unit of power and time is energy

power.compatible()      # ['W', 'Wp']

mass.u    # 'kg'
mass.v = # 5.0
```

You can do most of basic arithmetic operations (+,-,*,/) and relational operations (==, <, >, etc.) between variables. Only compatible units can be added or substracted (as far as I'm aware, `mass + power` wouldn't mean anything, right?). Multiplication and division follows conventional unit conversions. `float`s are assumed as non-dimensional values.

```python
import antupy as ap
time = Var(1, "day")
nom_power = Var(100, "kW")
energy = nom_power * time
energy_2 = (8*energy/2 - energy*2) # 200 ["kW-day"]

if energy >=  energy_2:
    print("energy is greater than energy_2")

```


### Thermophysical Properties
Here's an example of the helper module `props` to retrieve thermophysical properties of water.
```python
import antupy as ap

# Calculate energy stored in a water tank
temp_max = ap.Var(60, "degC")
temp_mains = ap.Var(20, "degC")
vol_tank = ap.Var(300, "L")

# Get temperature-dependent properties
fluid = ap.props.Water()
temp_avg = (temp_max + temp_mains) / 2
cp = fluid.cp(temp_avg)   # Specific heat [J/kg-K]
rho = fluid.rho(temp_avg)  # Density [kg/m3]

# Calculate stored energy
q_stg = vol_tank * rho * cp * (temp_max - temp_mains)
print(f"Energy stored: {q_stg.su('kWh'):.1f}")  # Output in kWh
```

This is the basic usage. For deeper usage and the use of the simulation classes such as `Plant` and `Parametric`, see the `documentation`.

So far, some research projects that have used antupy:
- [bdr_csp](https://github.com/DavidSaldivia/bdr_csp): A repository for csp simulations.
- [tm_solarshift](https://github.com/DavidSaldivia/tm_solarshift): A repository for domestic electric water heating systems (specifically for the Australian market).

## Data
Some basic data for Chile and Australia are included in this package.

[^1]: *mapuzugun* is the language of the Mapuche people, the main indigenous group in Chile. _antü_ (_antv_) means sun, but it also represents one of the main _pijan_ (spirits) in the Mapuche mythology. Here the word is used in its first literal meaning. The name was chosen because the first version of this software was written in Temuco, at the historic Mapuche heartland.