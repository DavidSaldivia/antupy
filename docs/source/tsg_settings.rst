Simulation Settings (tsg.settings)
====================================

The ``antupy.tsg.settings`` module provides time control and simulation parameter management for thermal and photovoltaic system simulations. The core component is the ``TimeParams`` class, which defines temporal boundaries and timestep parameters for all simulation scenarios.

Overview
--------

The settings module is designed to provide consistent time parameter management across all simulation workflows. It seamlessly integrates with both Polars and Pandas datetime systems, ensuring compatibility with modern data analysis tools while maintaining performance.

Key Features:

- **Flexible time control**: Define simulation start, stop, step, and year parameters
- **Dual datetime support**: Both Polars and Pandas datetime indexing
- **Property-based calculations**: Automatic derivation of simulation periods and days
- **Weather integration**: Native compatibility with weather generation systems
- **Unit-aware parameters**: All time parameters use antupy's Variable system with units

TimeParams Class
----------------

.. autoclass:: antupy.tsg.settings.TimeParams
   :members:
   :show-inheritance:
   :no-index:

The ``TimeParams`` class is the core time management system for simulations:

.. code-block:: python

   from antupy.tsg.settings import TimeParams
   from antupy import Var
   
   # Default annual simulation (full year with hourly timesteps)
   time_params = TimeParams()
   print(f"Default simulation: {time_params.DAYS.gv('day')} days")
   print(f"Total periods: {time_params.PERIODS.gv('-')}")
   
   # Custom short simulation (1 week with 30-minute timesteps)
   custom_params = TimeParams(
       START=Var(0, "hr"),      # Start at beginning of year
       STOP=Var(168, "hr"),     # 7 days * 24 hours
       STEP=Var(30, "min"),     # 30-minute intervals
       YEAR=Var(2023, "-")      # Year 2023
   )

Parameters
~~~~~~~~~~

The TimeParams class accepts four key parameters:

**START** (Variable)
   Initial time of the simulation in hours from the beginning of the year.
   
   - Default: ``Var(0, "hr")`` (beginning of year)
   - Annual simulations: typically 0
   - Representative day simulations: can be any hour of the year

**STOP** (Variable)
   Final time of the simulation in hours from the beginning of the year.
   
   - Default: ``Var(8760, "hr")`` (end of year for annual simulation)
   - Must be greater than START
   - For multi-day simulations: START + (days × 24)

**STEP** (Variable)
   Timestep interval for the simulation in minutes.
   
   - Default: ``Var(60, "min")`` (hourly timesteps)
   - Common values: 15, 30, 60 minutes
   - Higher resolution: shorter timesteps for detailed analysis

**YEAR** (Variable)
   Reference year for the simulation (dimensionless).
   
   - Default: ``Var(1800, "-")`` (placeholder year)
   - Important for weather data alignment
   - Affects leap year calculations

Properties
----------

TimeParams provides several computed properties for simulation management:

DAYS Property
~~~~~~~~~~~~~

.. code-block:: python

   time_params = TimeParams(START=Var(0, "hr"), STOP=Var(72, "hr"))
   simulation_days = time_params.DAYS
   print(f"Simulation length: {simulation_days.gv('day')} days")  # 3 days

PERIODS Property
~~~~~~~~~~~~~~~~

.. code-block:: python

   time_params = TimeParams(
       START=Var(0, "hr"),
       STOP=Var(48, "hr"),     # 2 days
       STEP=Var(15, "min")     # 15-minute intervals
   )
   total_periods = time_params.PERIODS
   print(f"Total timesteps: {total_periods.gv('-')}")  # 192 periods

DateTime Indexing
-----------------

TimeParams provides two datetime indexing systems for compatibility:

Polars Index (idx)
~~~~~~~~~~~~~~~~~~

.. autoproperty:: antupy.tsg.settings.TimeParams.idx
   :no-index:

For high-performance data analysis with Polars:

.. code-block:: python

   import polars as pl
   from antupy.tsg.settings import TimeParams
   from antupy import Var
   
   time_params = TimeParams(
       START=Var(0, "hr"),
       STOP=Var(48, "hr"),
       STEP=Var(30, "min"),
       YEAR=Var(2023, "-")
   )
   
   # Get Polars datetime series
   polars_index = time_params.idx
   print(f"Index type: {type(polars_index)}")  # polars.Series
   print(f"First timestamp: {polars_index[0]}")
   
   # Create Polars DataFrame with weather data
   weather_df = pl.DataFrame({
       "datetime": polars_index,
       "temperature": [20.5] * len(polars_index),
       "irradiance": [800.0] * len(polars_index)
   })

Pandas Index (idx_pd)
~~~~~~~~~~~~~~~~~~~~~

.. autoproperty:: antupy.tsg.settings.TimeParams.idx_pd
   :no-index:

For compatibility with Pandas-based workflows:

.. code-block:: python

   import pandas as pd
   from antupy.tsg.settings import TimeParams
   from antupy import Var
   
   time_params = TimeParams(
       START=Var(24, "hr"),     # Start from day 2
       STOP=Var(120, "hr"),     # 5 days total
       STEP=Var(60, "min"),     # Hourly
       YEAR=Var(2023, "-")
   )
   
   # Get Pandas datetime index
   pandas_index = time_params.idx_pd
   print(f"Index type: {type(pandas_index)}")  # pandas.DatetimeIndex
   print(f"Frequency: {pandas_index.freq}")
   
   # Create Pandas DataFrame
   weather_df = pd.DataFrame(
       index=pandas_index,
       data={
           "temp_amb": [25.0] * len(pandas_index),
           "GHI": [1000.0] * len(pandas_index)
       }
   )

Weather Integration
-------------------

TimeParams seamlessly integrates with the weather generation system:

.. code-block:: python

   from antupy.tsg.settings import TimeParams
   from antupy.tsg.weather import TMY, WeatherMC
   from antupy import Var
   
   # Define simulation timeframe
   summer_params = TimeParams(
       START=Var(6552, "hr"),   # Start of summer (approx. Dec 21)
       STOP=Var(6552 + 168, "hr"),  # One week
       STEP=Var(15, "min"),     # 15-minute resolution
       YEAR=Var(2023, "-")
   )
   
   # Create weather generators with time parameters
   tmy_weather = TMY(
       dataset="meteonorm",
       location="Sydney",
       time_params=summer_params
   )
   
   mc_weather = WeatherMC(
       dataset="meteonorm",
       location="Melbourne",
       time_params=summer_params,
       subset="season",
       value="summer",
       random=True
   )
   
   # Generate weather data aligned with time parameters
   tmy_data = tmy_weather.load_data()
   mc_data = mc_weather.load_data()
   
   # Both dataframes will have matching datetime indices
   assert len(tmy_data) == summer_params.PERIODS.gv('-')
   assert all(tmy_data.index == mc_data.index)

Common Usage Patterns
---------------------

Annual Simulations
~~~~~~~~~~~~~~~~~~

For full-year energy system analysis:

.. code-block:: python

   # Standard annual simulation
   annual_params = TimeParams(
       START=Var(0, "hr"),      # Beginning of year
       STOP=Var(8760, "hr"),    # End of year (365 days)
       STEP=Var(60, "min"),     # Hourly resolution
       YEAR=Var(2023, "-")      # Specific year
   )
   
   print(f"Annual simulation: {annual_params.DAYS.gv('day')} days")
   print(f"Total periods: {annual_params.PERIODS.gv('-')}")

Representative Day Simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For focused analysis on specific days:

.. code-block:: python

   # Summer design day (assuming day 355 = Dec 21)
   summer_day = TimeParams(
       START=Var(355 * 24, "hr"),      # December 21st
       STOP=Var(356 * 24, "hr"),       # December 22nd
       STEP=Var(10, "min"),            # High resolution
       YEAR=Var(2023, "-")
   )
   
   # Winter design day (assuming day 172 = June 21)
   winter_day = TimeParams(
       START=Var(172 * 24, "hr"),      # June 21st
       STOP=Var(173 * 24, "hr"),       # June 22nd
       STEP=Var(10, "min"),
       YEAR=Var(2023, "-")
   )

High-Resolution Simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For detailed transient analysis:

.. code-block:: python

   # One day with minute-level resolution
   detailed_params = TimeParams(
       START=Var(0, "hr"),
       STOP=Var(24, "hr"),      # Single day
       STEP=Var(1, "min"),      # Every minute
       YEAR=Var(2023, "-")
   )
   
   print(f"High resolution: {detailed_params.PERIODS.gv('-')} timesteps per day")

Multi-Day Studies
~~~~~~~~~~~~~~~~~

For week or month-long analysis:

.. code-block:: python

   # One week simulation
   weekly_params = TimeParams(
       START=Var(0, "hr"),
       STOP=Var(168, "hr"),     # 7 days
       STEP=Var(30, "min"),     # 30-minute intervals
       YEAR=Var(2023, "-")
   )
   
   # One month simulation (January)
   monthly_params = TimeParams(
       START=Var(0, "hr"),
       STOP=Var(31 * 24, "hr"), # 31 days
       STEP=Var(60, "min"),     # Hourly
       YEAR=Var(2023, "-")
   )

Advanced Features
-----------------

Unit Flexibility
~~~~~~~~~~~~~~~~

TimeParams leverages antupy's unit system for flexible parameter specification:

.. code-block:: python

   from antupy import Var
   
   # Different ways to specify the same 2-day simulation
   params1 = TimeParams(STOP=Var(48, "hr"))           # 48 hours
   params2 = TimeParams(STOP=Var(2880, "min"))        # 2880 minutes
   params3 = TimeParams(STOP=Var(2, "day"))           # 2 days (if supported)
   
   # All should give the same result
   assert params1.DAYS.gv('day') == 2
   # Note: Var unit conversion depends on antupy's unit system

Property Dependencies
~~~~~~~~~~~~~~~~~~~~~~

Properties automatically update when base parameters change:

.. code-block:: python

   params = TimeParams()
   original_periods = params.PERIODS.gv('-')
   
   # Modify timestep
   params.STEP = Var(30, "min")  # Change from 60 to 30 minutes
   new_periods = params.PERIODS.gv('-')
   
   # Periods should double due to smaller timestep
   assert new_periods == original_periods * 2

Performance Considerations
--------------------------

The TimeParams class provides both Polars and Pandas datetime indexing:

- **Use Polars (`idx`)** for new code and high-performance scenarios
- **Use Pandas (`idx_pd`)** for compatibility with existing pandas-based workflows
- **Properties are computed on-demand** - no overhead for unused calculations
- **Datetime generation is optimized** for large time series

Error Handling
--------------

TimeParams validates parameters and provides helpful error messages:

.. code-block:: python

   # This will raise appropriate errors:
   try:
       invalid_params = TimeParams(
           START=Var(100, "hr"),
           STOP=Var(50, "hr")      # STOP < START
       )
   except ValueError as e:
       print(f"Invalid time range: {e}")

Best Practices
--------------

1. **Match resolution to needs**: Use appropriate timesteps for your analysis requirements
2. **Consistent year specification**: Ensure YEAR aligns with weather data sources
3. **Memory considerations**: Large simulations with fine timesteps require more memory
4. **Weather integration**: Always use TimeParams with weather generators for consistency
5. **Unit awareness**: Leverage the Variable system for clear parameter specification

Integration Examples
--------------------

TimeParams works seamlessly with other antupy components:

.. code-block:: python

   from antupy.tsg.settings import TimeParams
   from antupy.tsg.weather import TMY
   from antupy import Var
   
   # Define simulation parameters
   sim_params = TimeParams(
       START=Var(0, "hr"),
       STOP=Var(8760, "hr"),
       STEP=Var(30, "min"),
       YEAR=Var(2023, "-")
   )
   
   # Use with weather generation
   weather = TMY(time_params=sim_params)
   weather_data = weather.load_data()
   
   # Weather data will have the correct datetime index
   assert len(weather_data) == sim_params.PERIODS.gv('-')
   assert weather_data.index.equals(sim_params.idx_pd)

See Also
--------

- :doc:`tsg_weather`: Weather generation system that uses TimeParams
- :doc:`variable_system`: Understanding the Variable (Var) system used for parameters
- :doc:`units`: Unit management and conversions
- ``antupy.core``: Core Variable and Array classes