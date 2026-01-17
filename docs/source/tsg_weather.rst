Time Series Generators (TSG)
=============================

The ``antupy.tsg`` module provides Protocol-based classes for generating time series data used in thermal system simulations. The main protocols are :py:class:`~antupy.tsg.weather.Weather` for environmental conditions and :py:class:`~antupy.tsg.mkt.Market` for electricity market prices. Both return pandas DataFrames with timezone-aware datetime indices.

Weather Classes
---------------

The weather module supports multiple data sources (METEONORM, MERRA2) and generation methods (TMY, Monte Carlo, historical, constant). All classes implement the :py:class:`~antupy.tsg.weather.Weather` Protocol and have a ``load_data()`` method that returns a DataFrame with columns like ``GHI``, ``temp_amb``, and ``temp_mains``.

Available weather classes:

- :py:class:`~antupy.tsg.weather.TMY` — Typical Meteorological Year data
- :py:class:`~antupy.tsg.weather.WeatherMC` — Monte Carlo sampling from datasets
- :py:class:`~antupy.tsg.weather.WeatherHist` — Historical weather from specific dates
- :py:class:`~antupy.tsg.weather.WeatherConstantDay` — Constant environmental conditions (this is mostly for testing purposes)

Example usage:

.. code-block:: python

    from antupy.tsg.weather import TMY, WeatherMC
    from antupy.tsg.settings import TimeParams
    from antupy.utils.loc import LocationAU
    
    # TMY weather for a location
    tp = TimeParams(start="2023-01-01", end="2023-12-31", step="1h")
    weather_tmy = TMY(
        dataset="meteonorm",
        location=LocationAU("Sydney"),
        time_params=tp
    )
    df_weather = weather_tmy.load_data()
    print(df_weather[["GHI", "temp_amb"]].head())
    
    # Monte Carlo sampling from summer season
    weather_mc = WeatherMC(
        dataset="meteonorm",
        location="Brisbane",
        time_params=tp,
        subset="season",
        value="summer",
        random=True
    )
    df_mc = weather_mc.load_data()

Market Classes
--------------

The market module provides electricity spot price data for different regions. All classes implement the :py:class:`~antupy.tsg.mkt.Market` Protocol and return DataFrames with a ``spot_price`` column in local currency.

Available market classes:

- :py:class:`~antupy.tsg.mkt.MarketAU` — Australian NEM spot prices by state
- :py:class:`~antupy.tsg.mkt.MarketCL` — Chilean SEN spot prices by barra (location)

Example usage:

.. code-block:: python

    from antupy.tsg.mkt import MarketAU, MarketCL
    
    # Australian market data
    market_au = MarketAU(
        state="NSW",
        year_i=2019,
        year_f=2019,
        dT=0.5  # half-hourly data
    )
    df_prices_au = market_au.load_data()
    print(df_prices_au["spot_price"].describe())
    
    # Chilean market data
    market_cl = MarketCL(
        location="crucero",
        year_i=2024,
        year_f=2024,
        dT=0.5
    )
    df_prices_cl = market_cl.load_data()

For complete API documentation including all parameters and methods, see the :doc:`api` reference.