API
===

Core Classes
------------

Unit-aware data structures for scalar values, arrays, and dataframes.

.. autosummary::
   :toctree: generated

   antupy.Var
   antupy.Array
   antupy.Frame
   antupy.CF
   antupy.Unit
   antupy.core.units.UnitDict


System Classes
--------------

Classes for modeling thermal systems and running parametric analyses.

.. autosummary::
   :toctree: generated

   antupy.Plant
   antupy.Simulation
   antupy.SimulationOutput
   antupy.Parametric


Time Series Generators (TSG)
-----------------------------

Weather data, Market data and other time series generation classes.

.. autosummary::
   :toctree: generated

   antupy.tsg.settings.TimeParams
   antupy.tsg.mkt.Market
   antupy.tsg.mkt.MarketAU
   antupy.tsg.mkt.MarketCL
   antupy.tsg.weather.Weather
   antupy.tsg.weather.TMY
   antupy.tsg.weather.WeatherMC
   antupy.tsg.weather.WeatherHist
   antupy.tsg.weather.WeatherConstantDay


Utilities
---------

Helper functions and classes for locations, heat transfer, thermophysical properties, and solar calculations.

.. autosummary::
   :toctree: generated

   antupy.utils.loc.Location
   antupy.utils.loc.LocationAU
   antupy.utils.loc.LocationCL


Thermophysical Properties
^^^^^^^^^^^^^^^^^^^^^^^^^

Material and fluid property classes.

.. autosummary::
   :toctree: generated

   antupy.utils.props.SolarSalt
   antupy.utils.props.Carbo
   antupy.utils.props.Aluminium
   antupy.utils.props.Copper
   antupy.utils.props.CopperNickel
   antupy.utils.props.StainlessSteel
   antupy.utils.props.Glass
   antupy.utils.props.SaturatedWater
   antupy.utils.props.SaturatedSteam
   antupy.utils.props.SeaWater
   antupy.utils.props.Air
   antupy.utils.props.HumidAir
   antupy.utils.props.CO2
   antupy.utils.props.TherminolVP1
   antupy.utils.props.Syltherm800
