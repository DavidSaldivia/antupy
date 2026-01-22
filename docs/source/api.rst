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

   antupy.loc.Location
   antupy.loc.LocationAU
   antupy.loc.LocationCL


Thermophysical Properties
^^^^^^^^^^^^^^^^^^^^^^^^^

**Available Materials** (provide ``rho(T)``, ``cp(T)``, ``k(T)`` methods):

.. autosummary::

   antupy.props.SolarSalt
   antupy.props.Carbo
   antupy.props.Aluminium
   antupy.props.Copper
   antupy.props.CopperNickel
   antupy.props.StainlessSteel
   antupy.props.Glass

**Available Fluids** (provide additional ``mu(T)``, ``h(T)``, ``Pr(T)`` and other fluid properties):

.. autosummary::

   antupy.props.SaturatedWater
   antupy.props.SaturatedSteam
   antupy.props.SeaWater
   antupy.props.Air
   antupy.props.HumidAir
   antupy.props.CO2
   antupy.props.TherminolVP1
   antupy.props.Syltherm800