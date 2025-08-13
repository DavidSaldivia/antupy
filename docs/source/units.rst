The unit manager
==================

The ``units`` module is the core of the unit manager used by Antupy. It contains the core class :py:class:`~antupy.units.Unit` that allows to represent any valid unit string into a standardized format. The standardized format, here called *base representantion* corresponds to a conversion factor (float) and a dictionary. The dictionary contains the unit name as key and the exponent as value.

The available units are classified in three categories: base units, derived units, and related units. The base units are the seven `SI base units <https://en.wikipedia.org/wiki/International_System_of_Units#SI_base_units>`_. The only difference is for mass, which is represented by gram (*g*), to simplify the task to represent prefixes. Additionally, the dimensionless quantity (represented by a single `-`), and the US dollars (USD), as base unit for money, are included, totalling nine base units.

Derived units are combinations of base units that represent other physical quantities, such as newton (N) for force, joule (J) for energy, and watt (W) for power. The SI unit system define `22 derived units <https://en.wikipedia.org/wiki/International_System_of_Units#Derived_units>`_ with special names and symbols that are included here.

Related units are commonly used units that are not part of the SI system but can be converted to SI units, such as liter (L) for volume, hectare (ha) for area, and bar for pressure. There are more than 30 related units available, which most likely will be extended in the future.

It is also possible to use any of the 22 SI prefixes to represent multiples or submultiples of the base units, such as kilo (k), mega (M), milli (m), and micro (μ). You can check the definitions of these units and prefixes in the ``units`` module:

.. code-block:: python

    from antupy import units
    print(units.BASE_UNITS)     # Output a dict with the definitions of base units
    print(units.DERIVED_UNITS)  # Output a dict with the definitions of derived units and its conversion to base units
    print(units.RELATED_UNITS)  # Output a dict with the definitions of related units
    print(units.PREFIXES)       # Output a dict with the definitions of SI prefixes

Valid unit strings
------------------

A valid unit string (here called a *unit label*) correspond to a combination of known units, connected by multiplication and/or division. The rules are very simple, and are defined to obtain a concise and unambiguous representation of the unit. They can be summarized as follows:

- Units can be multiplied by using the hyphon **(-)** symbol (e.g. ``m-s`` for meter-seconds). For simplicity, other symbols such as **\***, **x**, or simply whitespace are **not** allowed.
- Units can be raised to a power by appending the exponent to the unit (e.g. ``m2`` for square meters). Negative exponents are **not** allowed. Include them in the denominator.
- Units can be divided by using the **/** symbol (e.g. ``m/s`` for meters per second). **Only one** division symbol is allowed in a valid unit string.
- Parenthesis are **not** allowed yet.

Examples of valid and invalid unit strings:

.. code-block:: python

   from antupy.units import Unit
   # Valid unit strings
   Unit("m")
   Unit("m-s")
   Unit("km2")
   Unit("kJ")
   Unit("kg-m/s2")
   Unit("TW-hr")

   # Invalid unit strings (throw an error)
   Unit("m s")
   Unit("m*s")
   Unit("m/s2/s3")
   Unit("(m/s)")
   Unit("m-s-2")

The base representation
------------------------

The base representation of a unit is a tuple containing a conversion factor (a float, here called `base_factor`) and a dictionary with the unit names as keys and their respective exponents as values. This representation allows for easy conversion between units and is used internally by the :py:class:`~antupy.units.Unit` class. For example, the base representation of the unit "m/s2" is:

.. code-block:: python

    >>> from antupy.units import Unit
    >>> u = Unit("m/s2")
    >>> u.base_factor
    1.0
    >>> u.base_units
    {'s': -2, 'm': 1, 'g': 0, 'K': 0, 'A': 0, 'mol': 0, 'cd': 0, 'USD': 0}

Finally, you can show the SI representation by using the ``si`` property.

.. code-block:: python
    
    >>> from antupy.units import Unit
    >>> u = Unit("kJ")
    >>> u.label_unit
    'kJ'
    >>> u.si
    '1.00e+06[m2-g/s2]'


The ``units`` classes
-----------------

.. autoclass:: antupy.units.Unit
    :members:

.. autoclass:: antupy.units.UnitDict
    :members: