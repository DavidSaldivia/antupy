The ``antupy`` variable system
================================
``antupy`` works in its core with a Unit management module ``units``, which include the class ``Unit`` compatible with the SI unit system. From this, three type of variables are possible. The ``Var`` class to manage single variables in the form of ``(value:float, unit:str)`` structure. The ``Array`` class for structures in the form of ``(array:np.ndarray, unit:str)``.

The ``Var`` class is used to represent scalar values, while the ``Array`` class is used for vectors or time series data. Both classes are designed to handle units and conversions seamlessly, allowing for easy manipulation of physical quantities in simulations.

The ``Var`` class
-------------------
The ``Var`` class is a simple representation of a variable with a value and a unit. It allows for basic arithmetic operations, unit conversions, and comparisons. The class ensures that operations between variables are consistent in terms of units, providing a robust framework for handling physical quantities.

.. autoclass:: antupy.core.Var
    :members:


The ``Array`` class
-------------------
The ``Array`` class extends the functionality of the ``Var`` class to handle arrays of values, which can represent time series data or vectors. It supports operations such as element-wise arithmetic, unit conversions, and statistical analysis. The class is designed to work seamlessly with NumPy arrays, leveraging its powerful capabilities for numerical computations.

.. autoclass:: antupy.core.Array
    :members: