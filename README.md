# antupy
antupy (pronounced *antu-py*[^1], from the mapudungún word "antu" (sun)[^2]). It is an open-source python library to analyse (thermal) energy systems using stochastics methods. It includes a series of objects and methods to simulate energy conversion and energy storage systems, under uncertain timeseries constraints (weather, market, human behaviour, etc.).
It is an object-oriented software, with four main classes: Models, Timeseries Generators, Solvers, and Analysers. The different analysers allow a wide range of outputs such as: technical, economics (broader than just maximasing revenue), financial, (life-cycle) environmental impacts, etc.
Due to the wide range of possibilities, the current development is focused on applying this methodology on real research projects. Therefore, at the moment, the only Models implemented are domestic hot water heaters (DEWH) and concentrated solar thermal systems (CST). The available analysers are techno-economics and financial.

## methodology
`antupy` methodology divides the analysis in three sections: problem definition (pre-processing), simulations, and the analysis itself (post-processing).

### the problem definition
has two main outputs: simulation settings and timeseries generation. The first one deals with setting the numerical models for the simulations (things that don't change during the simulations), while the timeseries generation includes the uncertainties associated with monte-carlo simulation (things that change over time).

### simulations
key concepts:
    - MC simulations.
    - thermal simulation.
    - forecasts.


### analysis
The analysis is performed once the simulation is over.

## main classes
All these are Protocols.

### `Analysers`
They are the most fundamental objects. They define the type of analysis to be performed, and by consequence, the required models (and the interconnection between models), TSGs and Solvers to perform such analysis. 

### `Models`
represents real-world object that converts energy and/or mass flows following certain physical principle. Its functionality is defined by a protocol that includes: input/output flows, a numerical model (equations?) describing the input-to-output process, and a solver caller to simulate said model under certain inputs. A Model can contain other interconnected model.

### `Timeseries Generators (TSGs)`
These are objects that generate the timeseries for the simulations. Each one has a set of attributes and methods that fully define the timeseries generation algorithm. The most common ones are: Weather, Market, HWD, and ControlledLoad. 

### `Solvers`
Here is where the simulations are executed. Solvers can be own-made modules or wrappers of other (ideally open source) library/software. For example, for PV systems, we use pvlib, while for CSP both own methods or solarshift/SAM software are available.



## Examples
See the `examples` folder.

## Data


[^1]: IPA pronunciation.
[^2]: mapudungún is the language of the Mapuche people, the main indigineous group in Chile. _antü_ means sun, but also represents one of the main _pilláns_ (spirits) in Mapuche mythology. Here the word is used with its first literal meaning. The name was chosen because the first version of this library was written in Temuco, a Chilean city located at Mapuche heartland (*Wallmapu*). This library does not claim to include any of the Mapuche worldview in its development.