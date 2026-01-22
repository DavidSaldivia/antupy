# antupy
`antupy` (from the *mapuzugun* word "antü" (sun)[^1]) is an open-source python library to support the development of (solar thermal) energy research projects. It is a toolkit of classes and methods to help simulate energy conversion and energy storage systems, under uncertain timeseries constraints (weather, market, human behaviour, etc.).

An object-oriented software, it is structured in three main interdependent layers:
 - **Core layer**: A unit management system providing three classes to represent physical quantities with automatic unit tracking and conversion. `Var` for scalars, `Array` for vectors/timeseries, and `Frame` for tabular data with per-column units.
 - **Utilities layer**: A set of modules built on top of the core layer: `props` (thermophysical properties), `htc` (heat transfer correlations), `solar` (sun position and radiation calculations), and `loc` (geographical location management).
 - **Simulation layer**: Framework for building and analyzing energy systems using `Model`, `Plant`, and `Parametric` classes to support energy system simulation development (`Weather`, `Market`).

So far, some research projects that have used antupy:
- [bdr_csp](https://github.com/DavidSaldivia/bdr_csp): A repository for csp simulations.
- [tm_solarshift](https://github.com/DavidSaldivia/tm_solarshift): A repository for domestic electric water heating systems (specifically for the Australian market).

## Documentation
The full documentation is available [here](https://antupy.readthedocs.io/).

## Brief Introduction
`antupy` works in its core with a unit management module `units`, which include the class `Unit` to represent units that are compatible with the SI unit system. From this, three type of variables are introduced:
 1. The `Var` class to manage single variables, with the structure `(value:float, unit:str)`.
 2. The `Array` class for 1D data structures in the form of `(array:np.ndarray, unit:str)`.
 3. The `Frame` class for 2D data structures in the form of `(frame:pd.DataFrame, units:list[str])`.
Where the string `unit` (or `units`) has to follow a couple of [simple rules](https://antupy.readthedocs.io/en/latest/units.html#valid-unit-strings) to represent properly physical units. All three classes support arithmetic operations with automatic unit conversion and dimensional checking, ensuring dimensional consistency throughout calculations.

## Core Concepts

### `Plant` - System Integration Container
The `Plant` class is the main simulation container where you define components as class attributes and implement the `run_simulation()` method. Components can be dataclasses representing physical equipment with `run_model()` methods, TimeSeriesGenerators for weather/market data, or other model objects. The Plant orchestrates the simulation workflow and stores results in the `out` dictionary.

Example workflow:
1. Define components as class attributes (collectors, tanks, heat exchangers, etc.)
2. Implement `run_simulation()` to define the simulation logic
3. Access results through the `out` attribute

### `Parametric` - Sensitivity Analysis
The `Parametric` class enables parametric studies by running multiple simulations while systematically varying input parameters. It accepts a base `Plant` or `Simulation` instance and a dictionary of parameter ranges (as `Array` objects), runs all parameter combinations, and returns results as a `Frame` with units preserved. Supports detailed result saving and incremental CSV export for long-running studies.

### `TimeSeriesGenerator` (TSG)
Protocol for generating time-dependent boundary conditions. Current implementations include:
- `Weather`: TMY (Typical Meteorological Year), historical weather, Monte Carlo weather generation, and constant-day modes
- `Market`: Electricity price data for Australia (`MarketAU`) and Chile (`MarketCL`)

TSGs provide the `get_data()` method to retrieve timeseries data with proper unit tracking.

## Examples

### Quick Start - Core Classes
```python
from antupy import Var, Array, Frame
import numpy as np

# Scalar with units
mass = Var(5.0, "kg")
power = Var(100, "kW")
time = Var(1, "day")

# Automatic unit conversion and arithmetic
energy = (power * time).su("kWh")  # 2400 [kWh]
energy = (power * time).su("kW")   # Throws an error
print(f"Energy: {energy}")

# Arrays and Frames with units
temps = Array([20, 25, 30], "degC")

# Frames with per-column units
data = {"power": [10., 20., 40.], "area": [20., 35., 50.]}
df = Frame(data=data, units={"power": "MW", "area": "m2"})
```

### Thermophysical Properties
```python
from antupy import Var
from antupy.props import Water

# Calculate energy stored in a water tank
temp_max = Var(60, "degC")
temp_mains = Var(20, "degC")
vol_tank = Var(300, "L")
fluid = Water()

# Get temperature-dependent properties
temp_avg = (temp_max + temp_mains) / 2
cp = fluid.cp(temp_avg)   # Specific heat [J/kg-K]
rho = fluid.rho(temp_avg)  # Density [kg/m3]

# Calculate stored energy
q_stg = vol_tank * rho * cp * (temp_max - temp_mains)
print(f"Energy stored: {q_stg.su('kWh'):.1f}")  # Output in kWh
```

For complete examples including `Plant` simulations and `Parametric` studies, see the `documentation`.

## Data


[^1]: *mapuzugun* is the language of the Mapuche people, the main indigineous group in Chile. _antü_ (_antv_) means sun, but it also represents one of the main _pijan_ (spirits) in the Mapuche mythology. Here the word is used in its first literal meaning. The name was chosen because the first version of this software was written in Temuco, at the historic Mapuche heartland.