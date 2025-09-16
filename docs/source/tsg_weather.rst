Weather Generation (tsg.weather)
=================================

The ``antupy.tsg.weather`` module provides a comprehensive framework for weather data generation and management in thermal and photovoltaic simulations. It implements a Protocol-based architecture that supports multiple weather data sources and generation methods.

Overview
--------

The weather module is designed around a runtime-checkable ``Weather`` Protocol that ensures consistent interfaces across different weather data generation classes. This design allows for flexible weather simulation strategies while maintaining type safety and clear contracts.

Key Features:

- **Protocol-based architecture**: Runtime-checkable ``Weather`` Protocol for type safety
- **Multiple data sources**: Support for TMY, METEONORM, MERRA2, and historical datasets
- **Flexible generation methods**: TMY, Monte Carlo, historical, and constant day approaches
- **Australian location support**: Built-in integration with Australian weather stations
- **Time-aware simulations**: Full integration with ``TimeParams`` for temporal control

Weather Protocol
----------------

.. autoclass:: antupy.tsg.weather.Weather
   :members:
   :show-inheritance:
   :no-index:

The ``Weather`` Protocol defines the standard interface for all weather generators:

.. code-block:: python

   from antupy.tsg.weather import Weather, TMY
   from antupy.tsg.settings import TimeParams
   
   # Create a TMY weather generator
   weather = TMY(dataset="meteonorm", location="Sydney")
   
   # Runtime type checking
   assert isinstance(weather, Weather)  # True
   
   # Load weather data
   data = weather.load_data()

Weather Generator Classes
-------------------------

TMY (Typical Meteorological Year)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: antupy.tsg.weather.TMY
   :members:
   :show-inheritance:
   :no-index:

The TMY class generates weather data based on Typical Meteorological Year files, providing representative annual weather patterns:

.. code-block:: python

   from antupy.tsg.weather import TMY
   from antupy.tsg.settings import TimeParams
   from antupy.loc.loc_au import LocationAU
   
   # Create TMY weather with specific parameters
   weather = TMY(
       dataset="meteonorm",
       location=LocationAU("Melbourne"),
       time_params=TimeParams(start="2023-01-01", end="2023-12-31", step="1h")
   )
   
   # Generate weather data
   df_weather = weather.load_data()

WeatherMC (Monte Carlo)
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: antupy.tsg.weather.WeatherMC
   :members:
   :show-inheritance:
   :no-index:

The WeatherMC class provides Monte Carlo-based weather generation with random sampling capabilities:

.. code-block:: python

   from antupy.tsg.weather import WeatherMC
   
   # Monte Carlo weather with seasonal subset
   weather_mc = WeatherMC(
       dataset="meteonorm",
       location="Brisbane",
       subset="season",
       value="summer",
       random=True
   )
   
   # Generate random weather sample
   df_weather = weather_mc.load_data()

WeatherHist (Historical)
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: antupy.tsg.weather.WeatherHist
   :members:
   :show-inheritance:
   :no-index:

The WeatherHist class enables the use of specific historical weather data:

.. code-block:: python

   from antupy.tsg.weather import WeatherHist
   import pandas as pd
   
   # Historical weather for specific dates
   dates = pd.date_range("2022-06-01", "2022-06-30", freq="1h")
   weather_hist = WeatherHist(
       dataset="merra2",
       location="Perth",
       list_dates=dates,
       file_path="/path/to/weather/data.csv"
   )
   
   # Load historical data
   df_weather = weather_hist.load_data()

WeatherConstantDay
~~~~~~~~~~~~~~~~~~

.. autoclass:: antupy.tsg.weather.WeatherConstantDay
   :members:
   :show-inheritance:
   :no-index:

The WeatherConstantDay class provides constant environmental conditions throughout the simulation:

.. code-block:: python

   from antupy.tsg.weather import WeatherConstantDay
   
   # Constant weather conditions
   weather_const = WeatherConstantDay(
       location="Adelaide",
       random=False  # Use default constant values
   )
   
   # Generate constant weather data
   df_weather = weather_const.load_data()

Utility Functions
-----------------

The module provides several utility functions for weather data manipulation:

load_day_constant_random
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: antupy.tsg.weather.load_day_constant_random
   :no-index:

Generates random daily weather patterns within specified ranges:

.. code-block:: python

   import pandas as pd
   from antupy.tsg.weather import load_day_constant_random
   
   # Create empty timeseries
   ts = pd.DataFrame(
       index=pd.date_range("2023-01-01", periods=168, freq="1h"),
       columns=["GHI", "temp_amb", "temp_mains"]
   )
   
   # Fill with random daily values
   df_weather = load_day_constant_random(
       ts,
       ranges={
           "GHI": (800, 1200),
           "temp_amb": (15, 35),
           "temp_mains": (12, 25)
       },
       seed_id=42
   )

random_days_from_dataframe
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: antupy.tsg.weather.random_days_from_dataframe
   :no-index:

Randomly samples days from an existing weather dataset:

.. code-block:: python

   from antupy.tsg.weather import random_days_from_dataframe
   
   # Sample random days from historical data
   df_sampled = random_days_from_dataframe(
       timeseries=ts,
       df_sample=historical_weather,
       seed_id=123,
       columns=["GHI", "temp_amb"]
   )

Data Sources and Formats
------------------------

Supported Datasets
~~~~~~~~~~~~~~~~~~

The weather module supports multiple data sources:

- **METEONORM**: High-quality TMY data for global locations
- **MERRA2**: NASA's reanalysis dataset with historical weather
- **NCI**: National Computational Infrastructure weather data
- **Local files**: Custom CSV files with weather data

Standard Weather Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~

All weather generators work with standardized variable names:

- ``GHI``: Global Horizontal Irradiance (W/m²)
- ``temp_amb``: Ambient Temperature (°C)
- ``temp_mains``: Mains Water Temperature (°C)

Data Format
~~~~~~~~~~~

Weather data is returned as pandas DataFrames with:

- **Index**: DateTime index matching the specified time parameters
- **Columns**: Weather variables (GHI, temp_amb, temp_mains, etc.)
- **Values**: Numeric weather data in appropriate units

Location Integration
---------------------

The weather module integrates seamlessly with the location system:

.. code-block:: python

   from antupy.tsg.weather import TMY
   from antupy.loc.loc_au import LocationAU
   
   # Using LocationAU objects
   location = LocationAU("Sydney")
   weather = TMY(location=location)
   
   # Using string locations (automatically converted)
   weather = TMY(location="Melbourne")

Advanced Usage
--------------

Protocol-based Type Checking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Leverage runtime type checking for robust code:

.. code-block:: python

   from antupy.tsg.weather import Weather, TMY, WeatherMC
   
   def process_weather(weather_gen: Weather) -> pd.DataFrame:
       """Process any weather generator that implements the Weather protocol."""
       if not isinstance(weather_gen, Weather):
           raise TypeError("Expected Weather protocol implementation")
       return weather_gen.load_data()
   
   # Works with any weather generator
   tmy_data = process_weather(TMY())
   mc_data = process_weather(WeatherMC())

Custom Weather Generators
~~~~~~~~~~~~~~~~~~~~~~~~~

Create custom weather generators by implementing the Weather protocol:

.. code-block:: python

   from dataclasses import dataclass, field
   from antupy.tsg.weather import Weather
   from antupy.tsg.settings import TimeParams
   from antupy.loc import Location
   
   @dataclass
   class CustomWeather:
       dataset: str = "custom"
       location: str | Location = "Sydney"
       time_params: TimeParams = field(default_factory=TimeParams)
       
       def load_data(self) -> pd.DataFrame:
           # Custom implementation
           ts_index = self.time_params.idx_pd
           return pd.DataFrame(index=ts_index, data={"GHI": 1000, "temp_amb": 25})
   
   # Verify protocol compliance
   custom = CustomWeather()
   assert isinstance(custom, Weather)  # True

Error Handling
--------------

The weather module provides comprehensive error handling:

.. code-block:: python

   from antupy.tsg.weather import WeatherHist
   
   try:
       weather = WeatherHist(file_path="nonexistent.csv")
       data = weather.load_data()
   except FileNotFoundError:
       print("Weather file not found")
   except ValueError as e:
       print(f"Invalid weather data: {e}")

Best Practices
--------------

1. **Use Protocol typing**: Leverage the ``Weather`` Protocol for type hints and runtime checking
2. **Seed random generators**: Always provide ``seed_id`` for reproducible Monte Carlo simulations
3. **Validate locations**: Use ``LocationAU`` objects for Australian locations to ensure data availability
4. **Time parameter consistency**: Ensure ``TimeParams`` align with your simulation requirements
5. **Handle missing data**: Implement appropriate fallbacks for missing weather data

See Also
--------

- :doc:`variable_system`: Understanding Var and Array classes
- :doc:`units`: Unit management and conversions
- ``antupy.tsg.settings``: Time parameter configuration
- ``antupy.loc``: Location management system