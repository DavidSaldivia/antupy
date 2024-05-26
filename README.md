# antupy
antupy (pronounced *antu-py*[^1], from the mapudungún word "antu" (sun)[^2]). It is an open-source python library to analyse (thermal) energy systems using stochastics methods. It includes a series of objects and methods to simulate energy conversion and energy storage systems, under uncertain timeseries constraints (weather, market, human behaviour, etc.).
It is an object-oriented software, with four main classes: Models, Timeseries Generators, Solvers, and Analysers. The different analysers allow a wide range of outputs such as: technical, economics (broader than just maximasing revenue), financial, (life-cycle) environmental impacts, etc.
Due to the wide range of possibilities, the current development is focused on applying this methodology on real research projects. Therefore, at the moment, the only Models implemented are domestic hot water heaters (DEWH) and concentrated solar thermal systems (CST). The available analysers are techno-economics and financial.

## methodology
`antupy` methodology divides the analysis in three sections: problem definition (pre-processing), simulations, and the analysis itself (post-processing).

### the problem definition
has two main outputs: simulation settings and timeseries generation. The first one deals with setting the numerical models for the simulations (things that don't change during the simulations), while the timeseries generation includes the uncertainties associated with monte-carlo simulation (things that change over time).

### simulations

### analysis
The analysis is performed once the simulation is over.

## main classes
### The `Models` class
represents real-world object that converts energy and/or mass flows following certain physical principle. Its functionality is defined by a protocol that includes: input/output flows, a numerical model (equations?) describing the input-to-output process, and a solver caller to simulate said model under certain inputs. A Model can contain other interconnected model.

### `Timeseries Generators`

### `Solvers`

### `Analysers`


## Examples
See the `examples` folder.

## Data


[^1]: IPA pronunciation.
[^2]: mapudungún is the language of the Mapuche people, the main indigineous group in Chile. antü means sun, which also represents one of the main pilláns (spirits) in Mapuche mythology. Here the word is used with its first literal meaning. The name was chosen because the first version of this library was written in Temuco, a Chilean city located at Mapuche heartland (*Wallmapu*). More importantly, this library does not claim to include any of the Mapuche worldview in its development. I am Chilean (*huinca* from a mapuche point of view), although I do support Mapuche people's claims regarding their autonomy, lands, and self-determination, I do not consider myself mapuche.