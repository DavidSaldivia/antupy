"""
module with the core classes for AntuPy
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Self

import numpy as np

from antupy.units import Unit, _conv_temp, _mul_units, _div_units

def CF(unit1: str|Unit, unit2: str|Unit) -> Var:
    if isinstance(unit1, Unit):
        u1 = unit1
    else:
        u1 = Unit(unit1)    
    if isinstance(unit2, Unit):
        u2 = unit2
    else:
        u2 = Unit(unit2)
    if u1.base_units == u2.base_units:
        return Var(
            u1.base_factor / u2.base_factor,
            _div_units(u2.label_unit, u1.label_unit)
        )
    else:
        raise ValueError(f"{unit1} and {unit2} are not compatible.")

def _assign_unit(unit: str|Unit|None = None) -> Unit:
    if isinstance(unit, str):
        return Unit(unit)
    elif isinstance(unit, Unit):
        return unit
    else:
        raise TypeError(f"{type(unit)} is not a valid type for unit.")

@dataclass(frozen=True)
class Var():
    """
    Class to represent parameters and variables in the system.
    It is used to store the values with their units.
    If you have a Var instance, you can obtain the value in different units with the gv([str]) method.
    In this way you make sure you are getting the value with the expected unit.
    "gv" internally converts unit if it is possible.
    """
    value: float|None = None
    unit_: str|Unit|None = None
    unit: Unit = field(init=False)
    def __post_init__(self):
        object.__setattr__(self, "unit", _assign_unit(self.unit_))
    # def __init__(self, value: float|None=None, unit:str|Unit|None = None):
    #     self.value: float | None = value
    #     self.unit: Unit = _assign_unit(unit)

    def __add__(self, other: Self):
        """ Overloading the addition operator. """
        if not isinstance(other, Var):
            return NotImplemented
        if self.value is None or other.value is None:
            return Var(None, self.unit)
        if self.unit == other.unit:
            return Var(self.value + other.value, self.unit)
        elif self.unit.base_units == other.unit.base_units:
            return Var(self.value + other.gv(self.unit.label_unit), self.unit)
        else:
            raise TypeError(f"Cannot add {self.unit} with {other.unit}. Units are not compatible.")
        
    def __sub__(self, other: Self):
        """ Overloading the subtraction operator. """
        if not isinstance(other, Var):
            return NotImplemented
        if self.value is None or other.value is None:
            return Var(None, self.unit)
        if self.unit == other.unit:
            return Var(self.value - other.value, self.unit)
        elif self.unit.base_units == other.unit.base_units:
            return Var(self.value - other.gv(self.unit.u), self.unit)
        else:
            raise TypeError(f"Cannot subtract {self.unit} with {other.unit}. Units are not compatible.")

    def __radd__(self, other: Self):
        """ Overloading the addition operator. """
        if not isinstance(other, Var):
            return NotImplemented
        if self.value is None or other.value is None:
            return Var(None, self.unit)
        if self.unit == other.unit:
            return Var(self.value + other.value, other.unit)
        elif self.unit.base_units == other.unit.base_units:
            return Var(other.value + self.gv(other.unit.u), other.unit)
        else:
            raise TypeError(f"Cannot add {self.unit} with {other.unit}. Units are not compatible.")

    def __mul__(self, other: Self|float|int):
        """ Overloading the multiplication operator. """
        if isinstance(other, Var):
            if self.value is None or other.value is None:
                return Var(None, _mul_units(self.unit.u, other.unit.u))
            return Var(self.value * other.value, _mul_units(self.unit.u, other.unit.u))
        elif isinstance(other, (int, float)):
            if self.value is None:
                return Var(None, self.unit)
            return Var(self.value * other, self.unit)
        else:
            raise TypeError(f"Cannot multiply {type(self)} with {type(other)}")

    def __rmul__(self, other: Self|float|int):
        """ Overloading the multiplication operator. """
        if isinstance(other, Var):
            if self.value is None or other.value is None:
                return Var(None, _mul_units(other.unit.u, self.unit.u))
            return Var(self.value * other.value, _mul_units(other.unit.u, self.unit.u))
        elif isinstance(other, (int, float)):
            if self.value is None:
                return Var(None, self.unit)
            return Var(self.value * other, self.unit)
        else:
            raise TypeError(f"Cannot multiply {type(self)} with {type(other)}")

    def __truediv__(self, other: Self|float|int):
        """ Overloading the division operator. """
        if isinstance(other, Var):
            if self.value is None or other.value is None:
                return Var(None, _div_units(self.unit.u, other.unit.u))
            return Var(self.value / other.value, _div_units(self.unit.u, other.unit.u))
        elif isinstance(other, (int, float)):
            if self.value is None:
                return Var(None, self.unit)
            return Var(self.value / other, self.unit)
        else:
            raise TypeError(f"Cannot divide {type(self)} by {type(other)}")
        
    
        
    def __eq__(self, other) -> bool:
        """ Overloading the equality operator. """
        if not isinstance(other, Var) or other.value is None:
            return False
        return (
            self.value == other.value * CF(other.unit.u, self.unit.u).v
            and self.unit.base_units == other.unit.base_units
        )

    def __repr__(self) -> str:
        return f"{self.value:} [{self.unit.u}]"

    def get_value(self, unit: str | None = None) -> float:
        """ Method to obtain the value of the variable in the requested unit.
        If the unit is not compatible with the variable unit, an error is raised.
        If the unit is None, the value is returned in the Var's label unit.
        """
        if unit is None:
            unit = self.unit.u
        if self.value is None:
            raise ValueError("Var value is None.")
        if self.unit == unit:
            return self.value
        if self.unit.base_units == Unit(unit).base_units:
            if unit in ["°C", "degC","K"]:
                return float(_conv_temp(self, unit))
            return self.value * CF(self.unit.u, unit).v
        else:
            raise ValueError( f"Var unit ({self.unit}) and wanted unit ({unit}) are not compatible.")

    
    def set_unit(self, unit: str | None = None) -> Var:
        """ Set the primary unit of the variable. """
        unit = str(unit)
        if (self.unit.base_units == Unit(unit).base_units) and (self.value is not None):
            return Var(self.value * CF(self.unit, unit).v, Unit(unit))
        else:
            raise ValueError(
                f"unit ({unit}) is not compatible with existing unit label ({self.unit})."
            )

    @property
    def u(self) -> str:
        """ Property to obtain the label unit of the variable"""
        return self.unit.label_unit

    @property
    def v(self) -> float:
        """ Property to obtain the value of the variable in its label unit. """
        return self.value if self.value is not None else np.nan

    def gv(self, unit: str|None = None) -> float:
        """Alias for self.get_value()"""
        return self.get_value(unit)
    
    def su(self, unit: str|None = None) -> Var:
        """Alias of self.set_unit"""
        return self.set_unit(unit)

@dataclass(frozen=True)
class Array():
    """
    Class to represent data with the same units, such as timeseries, parametric analysis variables, etc.
    It is represented by the attributes values:np.ndarray and a unit:Unit.
    """
    value_: np.ndarray|list|None= None
    unit_: str|Unit|None = None

    value: np.ndarray = field(init=False)
    unit: Unit = field(init=False)
    def __post_init__(self):
        object.__setattr__(self, "value", np.array(self.value_))
        object.__setattr__(self, "unit", _assign_unit(self.unit_))
    # def __init__(
    #         self,
    #         v: np.ndarray|list|None=None,
    #         u: str|Unit|None = None
    #     ):
    #     self.value: np.ndarray = np.array(v)
    #     self.unit: Unit = _assign_unit(u)
    
    def __add__(self, other: Self):
        """ Overloading the addition operator. """
        if not isinstance(other, Array):
            return NotImplemented
        if self.unit == other.unit:
            return Array(self.value + other.value, self.unit)
        elif self.unit.base_units == other.unit.base_units:
            return Array(self.value + other.gv(self.unit.label_unit), self.unit)
        else:
            raise TypeError(f"Cannot add {self.unit} with {other.unit}. Units are not compatible.")
        
    def __sub__(self, other: Self):
        """ Overloading the subtraction operator. """
        if not isinstance(other, Array):
            return NotImplemented
        if self.unit == other.unit:
            return Array(self.value - other.value, self.unit)
        elif self.unit.base_units == other.unit.base_units:
            return Array(self.value - other.gv(self.unit.u), self.unit)
        else:
            raise TypeError(f"Cannot subtract {self.unit} with {other.unit}. Units are not compatible.")
    
    def __radd__(self, other: Self):
        """ Overloading the addition operator. """
        if not isinstance(other, Array):
            return NotImplemented
        if self.unit == other.unit:
            return Array(self.value + other.value, other.unit)
        elif self.unit.base_units == other.unit.base_units:
            return Array(other.value + self.gv(other.unit.u), other.unit)
        else:
            raise TypeError(f"Cannot add {self.unit} with {other.unit}. Units are not compatible.")    
    
    def __mul__(self, other: Self|float|int):
        """ Overloading the multiplication operator. """
        if isinstance(other, Array):
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
    
    def __rmul__(self, other: Self|float|int):
        """ Overloading the multiplication operator. """
        if isinstance(other, Array):
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

    def __truediv__(self, other: Self|float|int):
        """ Overloading the division operator. """
        if isinstance(other, Array):
            return Array(self.value / other.value, _div_units(self.unit.u, other.unit.u))
        elif isinstance(other, (int, float)):
            if self.value is None:
                return Var(None, self.unit)
            return Array(self.value / other, self.unit)
        else:
            raise TypeError(f"Cannot divide {type(self)} by {type(other)}")

    def __eq__(self, other) -> bool:
        """ Overloading the equality operator. """
        if not isinstance(other, Array) or other.value is None:
            return False
        return (
            np.allclose(self.value, other.value * CF(other.unit.u, self.unit.u).v)
            and self.unit.base_units == other.unit.base_units
        )

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

class Frame:
    pass


from collections.abc import Iterable
from typing import TypedDict

class Simulation():
    pass

class Output(TypedDict):
    pass

class Analyser():
    def get_simulation_instance(self, cases: Iterable) -> Simulation:
        return Simulation()
    def run_simulation(self) -> Output:
        return Output()

