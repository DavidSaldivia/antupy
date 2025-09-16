Location System (loc)
======================

The ``antupy.loc`` module provides a comprehensive location management system for geographical positioning in thermal and photovoltaic simulations. It implements a Protocol-based architecture supporting multiple location specification methods with particular strength in Australian geographical data.

Overview
--------

The location system is designed around a runtime-checkable ``Location`` Protocol that ensures consistent interfaces across different location implementations. This allows flexible location specification while maintaining type safety and integration with weather data sources.

Key Features:

- **Protocol-based architecture**: Runtime-checkable ``Location`` Protocol for type safety
- **Multiple input formats**: Support for city names, postcodes, and coordinates
- **Australian focus**: Comprehensive Australian postcode and city database
- **Coordinate conversion**: Automatic conversion between postcodes, cities, and coordinates
- **Weather integration**: Native compatibility with weather generation systems
- **Chilean support**: Basic Chilean location support via ``LocationCL``

Location Protocol
-----------------

.. autoclass:: antupy.loc.loc.Location
   :members:
   :show-inheritance:
   :no-index:

The ``Location`` Protocol defines the standard interface for all location classes:

.. code-block:: python

   from antupy.loc import Location, LocationAU
   
   # Create a location
   location = LocationAU("Sydney")
   
   # Runtime type checking
   assert isinstance(location, Location)  # True
   
   # All Location implementations provide string representation
   location_str = str(location)

Location Classes
----------------

LocationAU (Australian Locations)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: antupy.loc.loc_au.LocationAU
   :members:
   :show-inheritance:
   :no-index:

The ``LocationAU`` class provides comprehensive Australian location support with multiple input formats:

**City Names**

.. code-block:: python

   from antupy.loc import LocationAU
   
   # Major Australian cities
   sydney = LocationAU("Sydney")
   melbourne = LocationAU("Melbourne")
   brisbane = LocationAU("Brisbane")
   perth = LocationAU("Perth")
   adelaide = LocationAU("Adelaide")
   
   print(f"Sydney coordinates: {sydney.coords}")
   print(f"Sydney state: {sydney.state}")
   print(f"Sydney postcode: {sydney.postcode}")

**Postcode Input**

.. code-block:: python

   # Using Australian postcodes
   sydney_cbd = LocationAU(2000)      # Sydney CBD
   melbourne_cbd = LocationAU(3000)   # Melbourne CBD
   brisbane_cbd = LocationAU(4000)    # Brisbane CBD
   
   print(f"Postcode 2000: {sydney_cbd.coords}")
   print(f"State for 2000: {sydney_cbd.state}")

**Coordinate Input**

.. code-block:: python

   # Using longitude, latitude tuples
   custom_location = LocationAU((151.2093, -33.8688))  # Sydney Opera House
   
   print(f"Nearest postcode: {custom_location.postcode}")
   print(f"Nearest state: {custom_location.state}")

LocationCL (Chilean Locations)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: antupy.loc.loc_cl.LocationCL
   :members:
   :show-inheritance:
   :no-index:

Basic Chilean location support:

.. code-block:: python

   from antupy.loc import LocationCL
   
   # Chilean cities
   santiago = LocationCL("Santiago")
   valparaiso = LocationCL("Valparaíso")
   
   print(f"Chilean location: {santiago}")

LocationAU Properties
---------------------

The ``LocationAU`` class provides several computed properties for geographical data access:

Coordinates Property
~~~~~~~~~~~~~~~~~~~~

Returns longitude and latitude as a tuple:

.. code-block:: python

   location = LocationAU("Perth")
   lon, lat = location.coords
   print(f"Perth coordinates: {lon:.4f}°E, {lat:.4f}°S")
   
   # Individual coordinate access
   print(f"Longitude: {location.lon}")
   print(f"Latitude: {location.lat}")

State Property
~~~~~~~~~~~~~~

Returns the Australian state or territory:

.. code-block:: python

   locations = [
       LocationAU("Sydney"),     # NSW
       LocationAU("Melbourne"),  # VIC
       LocationAU("Brisbane"),   # QLD
       LocationAU("Perth"),      # WA
       LocationAU("Adelaide"),   # SA
       LocationAU("Hobart"),     # TAS
       LocationAU("Darwin"),     # NT
       LocationAU("Canberra")    # ACT
   ]
   
   for loc in locations:
       print(f"{loc}: {loc.state}")

Postcode Property
~~~~~~~~~~~~~~~~~

Returns the nearest Australian postcode:

.. code-block:: python

   # From city name
   sydney_postcode = LocationAU("Sydney").postcode
   print(f"Sydney postcode: {sydney_postcode}")
   
   # From coordinates
   coords_location = LocationAU((144.9631, -37.8136))  # Melbourne CBD
   nearest_postcode = coords_location.postcode
   print(f"Nearest postcode: {nearest_postcode}")

Utility Functions
-----------------

The LocationAU module provides utility functions for postcode and coordinate conversions:

_from_postcode Function
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: antupy.loc.loc_au._from_postcode
   :no-index:

Convert postcodes to geographical information:

.. code-block:: python

   from antupy.loc.loc_au import _from_postcode
   
   # Get state from postcode
   state = _from_postcode(2000, get="state")  # Returns "NSW"
   
   # Get coordinates from postcode
   coords = _from_postcode(3000, get="coords")  # Returns (lon, lat)
   
   print(f"Postcode 2000: {state}")
   print(f"Postcode 3000 coordinates: {coords}")

_from_coords Function
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: antupy.loc.loc_au._from_coords
   :no-index:

Convert coordinates to Australian postcode/state information:

.. code-block:: python

   from antupy.loc.loc_au import _from_coords
   
   # Sydney Opera House coordinates
   coords = (151.2153, -33.8570)
   
   # Find nearest postcode
   postcode = _from_coords(coords, get="postcode")
   
   # Find state
   state = _from_coords(coords, get="state")
   
   print(f"Coordinates {coords}")
   print(f"Nearest postcode: {postcode}")
   print(f"State: {state}")

Weather Integration
-------------------

Location classes integrate seamlessly with the weather generation system:

.. code-block:: python

   from antupy.loc import LocationAU
   from antupy.tsg.weather import TMY, WeatherMC
   from antupy.tsg.settings import TimeParams
   from antupy import Var
   
   # Create locations
   sydney = LocationAU("Sydney")
   melbourne_postcode = LocationAU(3000)
   custom_coords = LocationAU((153.0281, -27.4679))  # Brisbane coordinates
   
   # Create time parameters
   time_params = TimeParams(
       START=Var(0, "hr"),
       STOP=Var(168, "hr"),  # One week
       STEP=Var(60, "min"),
       YEAR=Var(2023, "-")
   )
   
   # Use locations with weather generators
   weather_sydney = TMY(location=sydney, time_params=time_params)
   weather_melbourne = WeatherMC(location=melbourne_postcode, time_params=time_params)
   weather_coords = TMY(location=custom_coords, time_params=time_params)
   
   # Generate weather data
   sydney_data = weather_sydney.load_data()
   melbourne_data = weather_melbourne.load_data()
   coords_data = weather_coords.load_data()

String Location Support
-----------------------

Weather generators also accept string locations for convenience:

.. code-block:: python

   from antupy.tsg.weather import TMY
   
   # String locations (automatically converted)
   tmy_string = TMY(location="Adelaide")  # Uses string directly
   tmy_object = TMY(location=LocationAU("Adelaide"))  # Uses LocationAU object
   
   # Both approaches work, but LocationAU provides more functionality
   print(f"String location: {tmy_string.location}")
   print(f"Object location: {tmy_object.location}")
   print(f"Object coordinates: {tmy_object.location.coords}")

Australian Data Coverage
------------------------

The LocationAU class leverages comprehensive Australian geographical databases:

Supported Cities
~~~~~~~~~~~~~~~~

Major Australian cities with weather data support:

.. code-block:: python

   # METEONORM supported cities (weather data available)
   major_cities = [
       "Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide",
       "Hobart", "Darwin", "Canberra", "Newcastle", "Wollongong",
       "Geelong", "Townsville", "Cairns", "Toowoomba", "Ballarat"
   ]
   
   for city in major_cities:
       location = LocationAU(city)
       print(f"{city}: {location.state}, {location.coords}")

Postcode Coverage
~~~~~~~~~~~~~~~~~

Complete Australian postcode database with ~17,000 entries:

.. code-block:: python

   # Sample of postcode ranges by state
   postcode_examples = {
       "NSW": [2000, 2010, 2050, 2100],  # Sydney area
       "VIC": [3000, 3006, 3141, 3181],  # Melbourne area
       "QLD": [4000, 4006, 4101, 4171],  # Brisbane area
       "WA": [6000, 6009, 6050, 6163],   # Perth area
       "SA": [5000, 5006, 5063, 5108],   # Adelaide area
       "TAS": [7000, 7004, 7010, 7050],  # Hobart area
       "NT": [800, 810, 820, 870],       # Darwin/Alice Springs
       "ACT": [2600, 2601, 2602, 2900]   # Canberra area
   }
   
   for state, postcodes in postcode_examples.items():
       for postcode in postcodes:
           location = LocationAU(postcode)
           print(f"Postcode {postcode}: {location.state}")

Advanced Usage
--------------

Protocol-based Type Checking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Leverage runtime type checking for robust code:

.. code-block:: python

   from antupy.loc import Location, LocationAU, LocationCL
   
   def process_location(location: Location) -> tuple[float, float]:
       """Process any location that implements the Location protocol."""
       if not isinstance(location, Location):
           raise TypeError("Expected Location protocol implementation")
       
       # For LocationAU, we can access coordinates
       if isinstance(location, LocationAU):
           return location.coords
       else:
           # For other location types, provide fallback
           return (0.0, 0.0)  # or raise NotImplementedError
   
   # Works with any Location implementation
   sydney_coords = process_location(LocationAU("Sydney"))
   chile_location = process_location(LocationCL("Santiago"))

Custom Location Classes
~~~~~~~~~~~~~~~~~~~~~~~

Create custom location classes by implementing the Location protocol:

.. code-block:: python

   from antupy.loc import Location
   
   class LocationUS:
       """Custom US location implementation."""
       
       def __init__(self, value: str):
           self.value = value
       
       def __str__(self) -> str:
           return f"US: {self.value}"
   
   # Verify protocol compliance
   us_location = LocationUS("New York")
   assert isinstance(us_location, Location)  # True

Coordinate System Details
~~~~~~~~~~~~~~~~~~~~~~~~~

LocationAU uses the WGS84 coordinate system:

.. code-block:: python

   location = LocationAU("Sydney")
   lon, lat = location.coords
   
   # Longitude: East positive (typical for Australia: 110° to 155°E)
   # Latitude: South negative (typical for Australia: -10° to -45°S)
   print(f"Sydney: {lon:.4f}°E, {abs(lat):.4f}°S")
   
   # For international use, remember latitude sign
   if lat < 0:
       hemisphere = "South"
   else:
       hemisphere = "North"
   print(f"Hemisphere: {hemisphere}")

Performance Considerations
--------------------------

The Location system is optimized for typical usage patterns:

- **City name lookup**: Fast dictionary-based lookup for major cities
- **Postcode conversion**: Efficient pandas-based operations on postcode database
- **Coordinate distance**: Euclidean distance calculation for nearest postcode/state
- **Caching**: Properties are computed on-demand but could benefit from caching for repeated access

Memory Usage
~~~~~~~~~~~~

.. code-block:: python

   # Postcode database is loaded on-demand
   # First postcode operation loads the full CSV (~2MB)
   location1 = LocationAU(2000)  # Loads database
   location2 = LocationAU(3000)  # Reuses loaded database
   
   # For memory-sensitive applications, consider pre-loading
   # or using city names when possible

Error Handling
--------------

The Location system provides comprehensive error handling:

.. code-block:: python

   # Invalid location types
   try:
       invalid_location = LocationAU(12.34)  # Float not supported
   except TypeError as e:
       print(f"Type error: {e}")
   
   # Invalid postcode
   try:
       location = LocationAU(99999)  # Non-existent postcode
       state = location.state
   except (KeyError, IndexError) as e:
       print(f"Postcode not found: {e}")
   
   # Missing city data
   try:
       location = LocationAU("UnknownCity")
       coords = location.coords
   except KeyError as e:
       print(f"City not found: {e}")

Best Practices
--------------

1. **Use LocationAU for Australian projects**: Provides complete geographical integration
2. **Prefer city names for major cities**: More readable and weather-data compatible
3. **Use postcodes for specific locations**: When precise positioning is required
4. **Use coordinates for custom locations**: When working outside predefined cities/postcodes
5. **Type hint with Location Protocol**: Enables flexible location handling
6. **Cache location objects**: For repeated use in simulations

Integration Examples
--------------------

Complete workflow example combining locations with weather and time parameters:

.. code-block:: python

   from antupy.loc import LocationAU
   from antupy.tsg.settings import TimeParams
   from antupy.tsg.weather import TMY, WeatherMC
   from antupy import Var
   
   # Define multiple locations
   locations = [
       LocationAU("Sydney"),      # City name
       LocationAU(3000),          # Melbourne CBD postcode
       LocationAU((153.0281, -27.4679))  # Brisbane coordinates
   ]
   
   # Define simulation time
   time_params = TimeParams(
       START=Var(0, "hr"),
       STOP=Var(8760, "hr"),  # Full year
       STEP=Var(60, "min"),
       YEAR=Var(2023, "-")
   )
   
   # Generate weather data for all locations
   weather_data = {}
   for location in locations:
       weather = TMY(
           dataset="meteonorm",
           location=location,
           time_params=time_params
       )
       weather_data[str(location)] = weather.load_data()
       
       print(f"Location: {location}")
       print(f"  Type: {location.input_type}")
       print(f"  State: {location.state}")
       print(f"  Coordinates: {location.coords}")
       print(f"  Weather data shape: {weather_data[str(location)].shape}")

See Also
--------

- :doc:`tsg_weather`: Weather generation system that uses Location classes
- :doc:`tsg_settings`: Time parameter management for simulations
- :doc:`variable_system`: Understanding the Variable system
- ``antupy.ddd_au``: Australian data definitions and constants