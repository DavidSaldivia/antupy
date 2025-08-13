The ``antupy`` variable system
================================
``antupy`` works in its core with a Unit management module ``units``, which include the class ``Unit`` mostly compatible with the SI unit system. From this, two classes are built. The ``Var`` class to manage single variables in the form of ``(value:float, unit:str)`` structure. The ``Array`` class for structures in the form of ``(array:np.ndarray, unit:str)``.

The ``Var`` class is used to represent scalar values, while the ``Array`` class is used for vectors or time series data. Both classes are designed to handle units and conversions seamlessly, allowing for easy manipulation of physical quantities in simulations.

The ``Var`` class
-------------------
The :py:class:`~antupy.core.Var` class is a simple representation of a variable with a value and a unit. It allows for basic arithmetic operations, unit conversions, and comparisons. The class ensures that operations between variables are consistent in terms of units, providing a robust framework for handling physical quantities.

.. autoclass:: antupy.Var
    :members:

Using the ``Var`` class
---------------------------
To create a variable, you can instantiate the ``Var`` class with a value and a unit. For example:

.. code-block:: python

    from antupy.core import Var
    mass = Var(5.0, "kg")
    pressure = Var(101325, "Pa")

Retrieving the properties in different units with ``get_value`` or simple ``gv`` and  provide a compatible unit as argument.

.. code-block:: python
    
    print(mass.get_value("g"))  # Outputs: 5000.0 '[g]'
    print(mass.gv("ton"))  # Outputs: 0.005 '[ton]'
    print(mass.gv("s"))   # Outputs: ValueError

You can perform arithmetic operations, like addition and subtraction, if both units represent the same quantities. The units will be automatically handled, and the result will be in the first variable's unit. For example:

.. code-block:: python

    result = mass + Var(500, "g")  # Adds 0.5 kg to 5 kg
    print(result)  # Outputs: 5.5 '[kg]'

You can multiply and divide variables with different units, and the resulting unit will be automatically calculated. For example:

.. code-block:: python

    time_sim = Var(1, "day")
    nom_power = Var(100, "kW")
    energy = nom_power * time_sim
    print(energy) # Outputs: 100 [kW-day]
    print(energy.gv("kW-hr")) # Outputs:  2400.0
    print(energy.gv("kWh")) # Outputs: 2400.0
    print(energy.gv("kJ")) # Outputs 8640000.0
    print((8*energy/2 - energy*2)) # Outputs: 200 ["kW-day"]
    print((8*energy/2 - energy*2).su("kWh")) # Outputs: 4800 ["kWh"]
    print((8*energy/2 - energy*2).su("m"))  # Outputs: Traceback (most recent call last): ...

Be careful while converting between temperature units. You can convert between Celsius and Kelvin when using ``get_value``. However, you cannot add or substract between them.  ``°F`` is not supported.

The ``CF`` function
---------------------

The module also provides a ``CF`` function, to provide useful conversion factors for common units. The function accepts two strings, and return a ``Var`` object with the corresponding conversion factor. For example, to convert from meters to kilometers:

.. code-block:: python

    from antupy.core import CF
    distance = Var(1500, "m")
    distance_km = distance * CF("m", "km")   # Applies the conversion factor
    print(distance_km)  # Outputs: 1.5 [km]

.. autofunction:: antupy.core.CF


The ``Array`` class
-------------------
The ``Array`` class extends the functionality of the ``Var`` class to handle arrays of values, which can represent time series data or vectors. It supports operations such as element-wise arithmetic, unit conversions, and statistical analysis. The class is designed to work seamlessly with NumPy arrays, leveraging its powerful capabilities for numerical computations.

.. autoclass:: antupy.core.Array
    :members: