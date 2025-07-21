# antupy
antupy (pronounced *antu-py*[^1], from the mapudungún word "antu" (sun)[^2]) is an open-source python library to analyse (thermal) energy systems using stochastics methods. It includes a series of classes and methods to simulate energy conversion and energy storage systems, under uncertain timeseries constraints (weather, market, human behaviour, etc.).
It is an object-oriented software, with four main classes: Models, Timeseries Generators, Solvers, and Analysers. The different analysers allow a wide range of outputs such as: technical, economics (broader than just maximasing revenue), financial, (life-cycle) environmental impacts, etc. It also include a toolbox with classes and functions like an unit management system, a thermophysical properties library, and a heat transfer coefficient library.
Due to the wide range of possibilities, the current development is focused on applying this methodology on real research projects. Therefore, at the moment, the only Models implemented are domestic hot water heaters (DEWH) and concentrated solar thermal systems (CST). The available analysers are techno-economics and financial.

## the `antupy` variable system
`antupy` works in its core with a Unit management module `units`, which include the class `Unit` compatible with the SI unit system. From this, three type of variables are possible. The `Var` class to manage single variables in the form of `(value:float, unit:str)` structure. The `Array` class for structures in the form of `(array:np.ndarray, unit:str)` and the `DataFrame` class for 2D-structures such as `(frame: pd.DataFrame|pl.DataFrame, unit:str)`.

## methodology
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


[^1]: IPA pronunciation.
[^2]: mapudungún is the language of the Mapuche people, the main indigineous group in Chile. _antü_ means sun, but it also represents one of the main _pilláns_ (spirits) in the Mapuche mythology. Here the word is used with its first literal meaning. The name was chosen because the first version of this library was written in Temuco, a Chilean city located at Mapuche heartland (*Wallmapu*). This library does not claim to include any of the Mapuche worldview in its development.
