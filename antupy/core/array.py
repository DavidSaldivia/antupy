from __future__ import annotations
from dataclasses import dataclass, field
from typing import Self

import numpy as np
from antupy.core.units import Unit, _conv_temp, _mul_units, _div_units, _assign_unit

from antupy.core.var import CF, Var

@dataclass(frozen=True)
class Array():
    """
    A container for homogeneous numerical data with associated physical units.
    
    The Array class represents collections of values (like time series, parametric 
    variables, or measurement arrays) that share the same physical unit. It provides 
    automatic unit conversion and arithmetic operations while preserving dimensional 
    consistency.

    Parameters
    ----------
    value_ : np.ndarray, list, or None, optional
        Input values as array-like or None. Default is None.
    unit_ : str, Unit, or None, optional
        Physical unit as string or Unit object. Default is None.

    Attributes
    ----------
    value : np.ndarray
        The numerical values as a numpy array.
    unit : Unit
        The physical unit associated with all values.

    Raises
    ------
    TypeError
        If units are incompatible during arithmetic operations.
    ValueError
        If requested unit conversion is not dimensionally consistent.

    Examples
    --------
    Basic usage:
    
    >>> from antupy import Array
    >>> temperatures = Array([20.0, 25.0, 30.0], "°C")
    >>> print(temperatures)
    [20. 25. 30.] [°C]

    Arithmetic operations with unit conversion:
    
    >>> mass1 = Array([1.0, 2.0, 3.0], "kg")
    >>> mass2 = Array([500, 1000, 1500], "g")
    >>> total_mass = mass1 + mass2  # Automatic conversion
    >>> print(total_mass)
    [1.5 2.5 3.5] [kg]

    Unit conversion:
    
    >>> print(total_mass.set_unit("g"))
    [1500. 2500. 3500.] [g]
    >>> print(total_mass.get_value("ton"))
    [0.0015 0.0025 0.0035]

    Iteration and indexing:
    
    >>> for temp in temperatures[:2]:
    ...     print(f"Temperature: {temp}")
    Temperature: 20.0 [°C]
    Temperature: 25.0 [°C]

    Notes
    -----
    All arithmetic operations automatically handle unit conversions when 
    dimensions are compatible. Incompatible operations raise TypeError.
    
    See Also
    --------
    Var : For handling single values with units
    antupy.units.Unit : The underlying unit representation class
    """
    _value: np.ndarray|list|Array|None= None
    _unit: str|Unit|None = None

    value: np.ndarray = field(init=False)
    unit: Unit = field(init=False)
    
    def __post_init__(self):
        if isinstance(self._value, Array) and self._unit is None:
            object.__setattr__(self, "value", self._value.value)
            object.__setattr__(self, "unit", self._value.unit)
        elif isinstance(self._value, Array) and self._unit is not None:
            unit_ = _assign_unit(self._unit)
            object.__setattr__(self, "value", self._value.gv(unit_.label_unit))
            object.__setattr__(self, "unit", unit_)
        else:
            object.__setattr__(self, "value", np.array(self._value))
            object.__setattr__(self, "unit", _assign_unit(self._unit))

    def __add__(self, other: Self|Var):
        """ Overloading the addition operator. """
        if not isinstance(other, (Array, Var)):
            return NotImplemented
        if self.unit == other.unit:
            return Array(self.value + other.value, self.unit)
        elif self.unit.base_units == other.unit.base_units:
            return Array(self.value + other.gv(self.unit.label_unit), self.unit)
        else:
            raise TypeError(f"Cannot add {self.unit} with {other.unit}. Units are not compatible.")
    
    def __radd__(self, other: Self|Var):
        """ Overloading the addition operator. """
        if not isinstance(other, (Var,Array)):
            return NotImplemented
        if self.unit == other.unit:
            return Array(self.value + other.value, other.unit)
        elif self.unit.base_units == other.unit.base_units:
            return Array(other.value + self.gv(other.unit.u), other.unit)
        else:
            raise TypeError(f"Cannot add {self.unit} with {other.unit}. Units are not compatible.")    
        
    def __sub__(self, other: Self|Var):
        """ Overloading the subtraction operator. """
        if not isinstance(other, (Array, Var)):
            return NotImplemented
        if self.unit == other.unit:
            return Array(self.value - other.value, self.unit)
        elif self.unit.base_units == other.unit.base_units:
            return Array(self.value - other.gv(self.unit.u), self.unit)
        else:
            raise TypeError(f"Cannot subtract {self.unit} with {other.unit}. Units are not compatible.")
    
    def __rsub__(self, other: Self|Var):
        """ Overloading the subtraction operator. """
        if not isinstance(other, (Array, Var)):
            return NotImplemented
        if self.unit == other.unit:
            return Array(other.value - self.value, self.unit)
        elif self.unit.base_units == other.unit.base_units:
            return Array(other.gv(self.unit.u) - self.value, self.unit)
        else:
            raise TypeError(f"Cannot subtract {self.unit} with {other.unit}. Units are not compatible.")
    
    def __mul__(self, other: Self|Var|float|int):
        """ Overloading the multiplication operator. """
        if isinstance(other, (Array, Var)):
            if self.value is None:
                return Array(None, _mul_units(self.unit.u, other.unit.u))
            return Array(
                self.value * other.value,
                _mul_units(self.unit.u, other.unit.u)
            )
        elif isinstance(other, (int, float)):
            return Array(self.value * other, self.unit)
        else:
            raise TypeError(f"Cannot multiply {type(self)} with {type(other)}")
    
    def __rmul__(self, other: Self|Var|float|int):
        """ Overloading the multiplication operator. """
        if isinstance(other, (Array, Var)):
            if self.value is None:
                return Array(None, _mul_units(other.unit.u, self.unit.u))
            return Array(
                other.value * self.value,
                _mul_units(other.unit.u, self.unit.u)
            )
        elif isinstance(other, (int, float)):
            return Array(self.value * other, self.unit)
        else:
            raise TypeError(f"Cannot multiply {type(self)} with {type(other)}")

    def __truediv__(self, other: Self|Var|float|int):
        """ Overloading the division operator. """
        if isinstance(other, (Array, Var)):
            return Array(self.value / other.value, _div_units(self.unit.u, other.unit.u))
        elif isinstance(other, (int, float)):
            if self.value is None:
                return Var(None, self.unit)
            return Array(self.value / other, self.unit)
        else:
            raise TypeError(f"Cannot divide {type(self)} by {type(other)}")
        
    def __rtruediv__(self, other: Self|Var|float|int):
        """ Overloading the division operator. """
        if isinstance(other, (Array, Var)):
            return Array(other.value / self.value, _div_units(other.unit.u, self.unit.u))
        elif isinstance(other, (int, float)):
            if self.value is None:
                return Var(None, _div_units("", self.unit.u))
            return Array(other / self.value, _div_units("", self.unit.u))
        else:
            raise TypeError(f"Cannot divide {type(other)} by {type(self)}")

    def __eq__(self, other) -> bool:
        """ Overloading the equality operator. """
        if not isinstance(other, Array) or other._value is None:
            return False
        return (
            np.allclose(self.value, other.value * CF(other.unit.u, self.unit.u).v)
            and self.unit.base_units == other.unit.base_units
        )
    
    def __len__(self) -> int:
        return len(self.v)
    
    def __getitem__(self, key) -> Var:
        return Var(float(self.value[key]), self.u)
    
    def __iter__(self):
        return (Var(v, self.u) for v in self.value)

    def __repr__(self) -> str:
        return f"{self.value:} [{self.unit.u}]"

    def get_value(self, unit: str | None = None) -> np.ndarray:
        """ Method to obtain the value of the variable in the requested unit.
        If the unit is not compatible with the variable unit, an error is raised.
        If the unit is None, the value is returned in the variable unit.
        """
        if unit is None:
            unit = self.unit.u
        if self.unit == unit:
            return self.value
        if self.unit.base_units == Unit(unit).base_units:
            if unit in ["°C", "degC","K"]:
                return np.array(_conv_temp(self, unit))
            return self.value * CF(self.unit.u, unit).v
        else:
            raise ValueError( f"Var unit ({self.unit}) and wanted unit ({unit}) are not compatible.")

    def set_unit(self, unit: str | None = None) -> Array:
        """ Set the primary unit of the variable. """
        unit = str(unit)
        if (self.unit.base_units == Unit(unit).base_units) and (self.value is not None):
            return Array(self.value * CF(self.unit, unit).v, Unit(unit))
        else:
            raise ValueError(
                f"unit ({unit}) is not compatible with existing primary unit ({self.unit})."
            )

    @property
    def u(self) -> str:
        """ Property to obtain the label unit of the variable"""
        return self.unit.label_unit

    @property
    def v(self) -> np.ndarray:
        """ Property to obtain the value of the variable in its label unit. """
        return self.value

    def gv(self, unit:str|None = None) -> np.ndarray:
        """Alias for self.get_value()"""
        return self.get_value(unit)
    
    def su(self, unit: str|None = None) -> Array:
        """Alias of self.set_unit"""
        return self.set_unit(unit)
    
    def compatible(self) -> list[str]:
        """ Return a list of compatible units for the variable unit. """
        return self.unit.compatible()
    
    def mean(self, unit: str | None = None) -> Var:
        u = self.u if unit is None else unit
        return Var(self.gv(u).mean(), u)
    
    def std(self, unit: str | None = None) -> Var:
        u = self.u if unit is None else unit
        return Var(self.gv(u).std(), u)
    
    def var(self, unit: str | None = None) -> Var:
        u = self.u if unit is None else unit
        return Var(self.gv(u).var(), _mul_units(u, u))
    
    def max(self, unit: str | None = None) -> Var:
        u = self.u if unit is None else unit
        return Var(self.gv(u).max(), u)
    
    def min(self, unit: str | None = None) -> Var:
        u = self.u if unit is None else unit
        return Var(self.gv(u).min(), u)
    
    def argmax(self) -> int:
        return int(np.argmax(self.v))
    
    def argmin(self) -> int:
        return int(np.argmin(self.v))

    def sum(self, unit: str | None = None) -> Var:
        u = self.u if unit is None else unit
        return Var(self.gv(u).sum(), u)
    
    def prod(self, unit: str | None = None) -> Var:
        u = self.u if unit is None else unit
        u_f = ""
        for _ in range(len(self)):
            u_f = _mul_units(u, u_f)
        return Var(self.gv(u).prod(), u_f)
    
    def cumsum(self, unit: str | None = None) -> Array:
        u = self.u if unit is None else unit
        return Array(self.gv(u).cumsum(), u)
    
    def sort(self, unit: str | None = None) -> Array:
        u = self.u if unit is None else unit
        return Array(self.gv(u).sort(), u)
    
    def round(self, decimals: int = 0, unit: str | None = None) -> Array:
        u = self.u if unit is None else unit
        return Array(np.round(self.gv(u), decimals), u)