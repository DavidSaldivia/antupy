# antupy
`antupy` (from the *mapuzugun* word "antü" (sun)[^1]) is an open-source python library to analyse (solar thermal) energy systems. It is a toolkit of classes and methods to help simulate energy conversion and energy storage systems, under uncertain timeseries constraints (weather, market, human behaviour, etc.).

It is an object-oriented software, structured in three main interdependent modules:
 - A **unit manager** to store and represent physical quantities. It offers three classes to represent data structures. `Var`, for scalar or 0D data; `Array`, for 1D-vectors (or timeseries), and `Frame` for 2D-dataframe-style data.
 - To help **simulate** real world systems, it provides classes to represent `Model`s and `Plant`s, that can be used with different analysers, such as, the `Parametric` class. To help these simulations, a couple of Timeseries Generators (`TSG`s) are presented, such as `Weather`, `Market`, and Hot Water Draw (`HWD`) data generators. 
 - A set of **utility** modules based on the unit manager system. `props`: a thermophysical properties library; `htc`: a heat transfer coefficient library; `solar`: a module with solar-related functions; `loc`: a location manager.

Due to the wide range of possibilities and variety of these systems, the development has been focused on the developers' research. Therefore, at the moment, the only simulations implemented so far are domestic hot water heaters (DEWH) and concentrated solar thermal systems (CST).

## Documentation
Check the documentation, cloning the repository and run 'docs/make.bat html' in your python environment. The docs has not yet been published.

## The `antupy` variable system
`antupy` works in its core with a unit management module `units`, which include the class `Unit` to represent units that are compatible with the SI unit system. From this, three type of variables are introduced.
 1. The `Var` class to manage single variables in the form of `(value:float, unit:str)` structure.
 2. The `Array` class for structures in the form of `(array:np.ndarray, unit:str)`.
 3. The `Frame` class for structures in the form of `(frame:pd.DataFrame|pl.DataFrame, unit:str)`.

## Methodology
`antupy` methodology divides the analysis in three sections: problem definition (pre-processing), simulations, and the analysis itself (post-processing).

### the problem definition
has two main outputs: simulation settings and timeseries generation. The first one deals with things that don't change during the simulations, it is the numerical models and fixed parameters; while the timeseries generation includes the things that change during the simulation (weather, market data, etc.) including the possible uncertainties.

### simulations
key concepts:
    - thermal simulation.
    - energy system simulations.
    - MC simulations.
    - forecasts.


### analysis
The analysis is performed once the simulation is over.

## main classes
All these are Protocols.

### `Analysers`
They are the most fundamental objects. They define the type of analysis to be performed, and by consequence, the required models (and the interconnection between models), TSGs and Solvers to perform such analysis. 

### `Timeseries Generators (TSGs)`
These are objects that generate the timeseries for the simulations. Each one has a set of attributes and methods that fully define the timeseries generation algorithm. The most common ones are: Weather, Market, HWD, and ControlledLoad. 

### `Models`
represents real-world object that converts energy and/or mass flows following certain physical principle. Its functionality is defined by a protocol that includes: input/output flows, a numerical model (equations?) describing the input-to-output process, and a solver caller to simulate said model under certain inputs. A Model can contain other interconnected model.

### `Solvers`
Here is where the simulations are executed. Solvers can be own-made modules or wrappers of other (ideally open source) library/software. For example, for PV systems, we use pvlib, while for CSP both own methods or solarshift/SAM software are available.



## Examples
See the `examples` folder.

## Data


[^1]: *mapudungún* is the language of the Mapuche people, the main indigineous group in Chile. _antü_ means sun, but it also represents one of the main _pilláns_ (spirits) in the Mapuche mythology. Here the word is used in its first literal meaning. The name was chosen because the first version of this software was written in Temuco, a Chilean city located at the historic Mapuche heartland.