from __future__ import annotations
from dataclasses import dataclass, field
from typing import Self

import math

import numpy as np
from antupy.core.units import Unit, _assign_unit, _conv_temp, _mul_units, _div_units


def CF(unit1: str|Unit, unit2: str|Unit) -> Var:
    """
    Conversion factor between two units.
    It returns a Var instance with the conversion factor and the unit label.
    If the units are not compatible, it raises a ValueError.
    
    This function computes the multiplicative factor needed to convert a value
    from unit1 to unit2. The result is returned as a Var object containing
    the conversion factor and a compound unit representing the ratio.
    
    Parameters
    ----------
    unit1 : str or Unit
        The source unit (unit to convert from). Can be a unit string 
        (e.g., "m", "kg/s") or a Unit object.
    unit2 : str or Unit  
        The target unit (unit to convert to). Can be a unit string
        (e.g., "km", "g/s") or a Unit object.
    
    Returns
    -------
    Var
        A Var object where:
        - value: the numerical conversion factor
        - unit: compound unit representing unit2/unit1
    
    Raises
    ------
    ValueError
        If the units are not dimensionally compatible (e.g., trying to
        convert between length and mass units).
    TypeError
        If unit1 or unit2 are not valid unit types.
    
    Examples
    --------
    Basic unit conversions:
    
    >>> from antupy.core import CF
    >>> cf = CF("m", "km")
    >>> print(cf.v)  # Get the numerical value
    0.001
    >>> print(cf)    # Show the full conversion factor
    0.001 [km/m]
    
    Energy unit conversion:
    
    >>> cf_energy = CF("J", "kWh")
    >>> print(cf_energy)
    2.7777777777777776e-07 [kWh/J]
    
    Flow rate conversion:
    
    >>> cf_flow = CF("m3/s", "L/min")
    >>> print(cf_flow)
    60000.0 [L-min/m3-s]
    
    Usage in unit conversion:
    
    >>> # Convert 5 meters to kilometers
    >>> distance_m = 5000  # meters
    >>> distance_km = distance_m * CF("m", "km").v
    >>> print(f"{distance_m} m = {distance_km} km")
    5000 m = 5.0 km
    
    Notes
    -----
    The conversion factor represents how many units of unit2 equal one unit
    of unit1. For example, CF("m", "km") returns 0.001 because 1 meter = 
    0.001 kilometers.
    
    The function only works with dimensionally compatible units. For example,
    you cannot convert between "kg" (mass) and "m" (length).
    
    See Also
    --------
    Var.gv : Get value in different units
    antupy.units.Unit : The underlying unit representation
    """
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


@dataclass(frozen=True)
class Var():
    """
    Class to represent parameters and variables in the system.
    It is used to store the values with their units.
    If you have a Var instance, you can obtain the value in different units with the gv([str]) method.
    In this way you make sure you are getting the value with the expected unit.
    "gv" internally converts unit if it is possible.
    
    Parameters
    ----------
    value : float or None, optional
        The numerical value of the variable. If None, represents an undefined
        or uninitialized variable. Default is None.
    _unit : str, Unit, or None, optional
        The unit of the variable. Can be a unit string (e.g., "kg", "m/s2"),
        a Unit object, or None for dimensionless quantities. Default is None.
    
    unit : Unit
        The Unit object representing the variable's unit.
    
    Examples
    --------
    Creating variables with units:
    
    >>> from antupy.core import Var
    >>> mass = Var(5.0, "kg")
    >>> velocity = Var(10, "m/s")
    
    Arithmetic operations with automatic unit handling:
    
    >>> v1 = Var(5.0, "kg")
    >>> v2 = Var(500, "g")
    >>> total_mass = v1 + v2  # Automatically converts g to kg
    >>> print(total_mass)
    5.5 [kg]
    
    Unit conversions:
    
    >>> energy = Var(1000, "J")
    >>> energy_in_kj = energy.gv("kJ")
    >>> print(energy_in_kj)
    1.0
    
    Physical calculations:
    
    >>> force = mass * Var(9.81, "m/s2")  # F = ma
    >>> print(force)
    49.05 [kg-m/s2]
    
    Notes
    -----
    - The class is immutable (frozen dataclass) to ensure variable integrity
    - Operations between incompatible units raise TypeError
    - Supports comparison operators with automatic unit conversion
    - Can be converted to float/int for numerical operations
    
    See Also
    --------
    Array : For handling arrays of values with the same unit
    antupy.units.Unit : The underlying unit representation class
    """
    value: float|None = None
    _unit: str|Unit|None = None
    unit: Unit = field(init=False)
    def __post_init__(self):
        object.__setattr__(self, "unit", _assign_unit(self._unit))

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
            return NotImplemented

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
            return NotImplemented

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
            return NotImplemented
    
    def __int__(self) -> int:
        return int(self.v)

    def __float__(self) -> float:
        return float(self.v)

    def __eq__(self, other) -> bool:
        """ Overloading the equality operator. """
        if not isinstance(other, Var):
            return NotImplemented
        if other.value is None:
            return False
        return (
            self.value == other.value * CF(other.unit.u, self.unit.u).v
            and self.unit.base_units == other.unit.base_units
        )

    def __lt__(self, other) -> bool:
        if isinstance(other, Var):
            return self.v < other.gv(self.unit.u)
        elif isinstance(other, (int,float)):
            return self.v < other
        else:
            return NotImplemented
        
    def __le__(self, other) -> bool:
        if isinstance(other, Var):
            return self.v <= other.gv(self.unit.u)
        elif isinstance(other, (int,float)):
            return self.v <= other
        else:
            return NotImplemented
        
    def __gt__(self, other) -> bool:
        if isinstance(other, Var):
            return self.v > other.gv(self.unit.u)
        elif isinstance(other, (int,float)):
            return self.v > other
        else:
            return NotImplemented
        
    def __ge__(self, other) -> bool:
        if isinstance(other, Var):
            return self.v >= other.gv(self.unit.u)
        elif isinstance(other, (int,float)):
            return self.v >= other
        else:
            return NotImplemented
    
    def __neg__(self) -> Var:
        return Var(-self.value if self.value is not None else None, self.unit)
    
    def __pos__(self) -> Var:
        return Var(+self.value if self.value is not None else None, self.unit)
    
    def __abs__(self) -> Var:
        return Var(abs(self.value) if self.value is not None else None, self.unit)
    
    def __round__(self, ndigits=0) -> Var:
        return Var(round(self.value, ndigits) if self.value is not None else None, self.unit)

    def __trunc__(self) -> Var:
        return Var(math.trunc(self.value) if self.value is not None else None, self.unit)

    def __floor__(self) -> Var:
        return Var(math.floor(self.value) if self.value is not None else None, self.unit)

    def __ceil__(self) -> Var:
        return Var(math.ceil(self.value) if self.value is not None else None, self.unit)

    def __repr__(self) -> str:
        return f"{self.value:} [{self.unit.u}]"

    def __format__(self, format_spec: str) -> str:
        """
        Format the Var object for use in f-strings and format() calls.
        
        Supports various format specifications:
        - Precision: .2f, .3e, etc. (applies to value only)
        - Width/alignment: >15, <10, ^20 (applies to entire string)
        - Combined: >15.2f (width applies to entire string, precision to value)
        
        Parameters
        ----------
        format_spec : str
            Format specification string
            
        Returns
        -------
        str
            Formatted string representation of the Var
            
        Examples
        --------
        >>> var = Var(3.14159, "m")
        >>> f"{var:.2f}"
        '3.14 [m]'
        >>> f"{var:>15}"
        '    3.14159 [m]'
        >>> f"{var:>15.2f}"
        '       3.14 [m]'
        """
        if self.value is None:
            base_str = f"None [{self.unit.u}]"
            if format_spec:
                # Extract width/alignment only for None values
                import re
                width_match = re.match(r'([<>=^]?)(\d+)', format_spec)
                if width_match:
                    align, width = width_match.groups()
                    return format(base_str, f"{align}{width}")
            return base_str
        
        # Parse format specification to separate width/alignment from precision/type
        import re
        
        # Match format patterns like: [fill][align][width][.precision][type]
        match = re.match(r'([<>=^]?)(\d*)(?:\.(\d+))?([a-zA-Z%]?)', format_spec)
        if match:
            align, width, precision, type_spec = match.groups()
            
            # Build format for the numeric value
            value_format = ""
            if precision:
                value_format += f".{precision}"
            if type_spec:
                value_format += type_spec
            
            # Format the value
            if value_format:
                formatted_value = format(self.value, value_format)
            else:
                formatted_value = str(self.value)
            
            # Create the full string
            base_str = f"{formatted_value} [{self.unit.u}]"
            
            # Apply width/alignment to the full string
            if width:
                return format(base_str, f"{align}{width}")
            else:
                return base_str
        else:
            # Fallback for unrecognized format specs
            return f"{self.value} [{self.unit.u}]"

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

