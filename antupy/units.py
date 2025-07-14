"""
module with a simple units manager
"""
import numpy as np
from typing import Iterable, Self
from dataclasses import dataclass, field

CONVERSION_FUNDAMENTALS: dict[str,dict[str|None,float]] = {
    "adim" : {
        "-": 1e0,
        "": 1e0,
        " ": 1e0,
        "(-)": 1e0,
        "()": 1e0,
    },
    "length" : {
        "m": 1e0,
        "mm": 1e3,
        "km": 1e-3,
        "mi": 1e0/1609.34,
        "ft": 3.28084,
        "in": 39.3701,
    },
    "mass": {
        "kg": 1e0,
        "g": 1e3,
        "ton": 1e-3,
        "lb": 2.20462,
        "oz": 35.274,
    },
    "time": {
        "s": 1e0, "sec": 1e0,
        "min": 1e0/60,
        "h": 1e0/3600, "hr": 1e0/3600,
        "d": 1e0/(24*3600), "day": 1e0/(24*3600),
        "wk": 1e0/(24*3600*7), "week": 1e0/(24*3600*7),
        "mo": 1e0/(24*3600*30), "month": 1e0/(24*3600*30),
        "yr": 1e0/(24*3600*365), "year": 1e0/(24*3600*365),
    },
    "temperature": {
        "K": 1.0,
        "C": np.nan
    },
    "current": {
        "A": 1e0,
        "mA": 1e3,
        "kA": 1e-3,
    },
    "substance": {
        "mol": 1e0,
        "mmol": 1e3,
        "kmol": 1e-3,
    },
    "luminous_intensity": {
        "cd": 1e0,
        "lm": 1e0,
    },
}

CONVERSIONS_DERIVED: dict[str,dict[str|None,float]] = {
    "area" : {
        "m2": 1e0,
        "mm2": 1e6,
        "km2": 1e-6,
        "ha": 1e-4,
    },
    "volume": {
        "m3": 1e0,
        "L": 1e3,
        },
    "mass_flowrate": {
        "kg/s": 1e0,
        "g/s": 1e3,
        "kg/min": 60,
        "kg/hr": 3600,
    },
    "volume_flowrate": {
        "L/s": 1e0,
        "m3/s": 1e-3,
        "m3/min": 1e-3*60,
        "m3/hr": 1e-3*3600,
        "L/min": 60,
        "L/hr": 3600,
        "ml/s": 1e3,
    },
    "energy": {
        "J": 1e0,
        "kJ": 1e-3,
        "MJ": 1e-6,
        "Wh": 1e-3/3.6,
        "kWh": 1e-6/3.6,
        "MWh": 1e-9/3.6,
        "cal": 4.184e0,
        "kcal": 4.184e3,
    },
    "energy_flow": {
        "MW/m2": 1e0,
        "kJ/m2": 1e3,
        "J/m2": 1e6,
    },
    "power": {
        "W": 1e0,
        "kW": 1e-3,
        "MW": 1e-6,
        "J/h": 3.6e6, "J/hr": 3.6e6,
        "kJ/h": 3.6e0, "kJ/hr": 3.6e0,
        "MJ/h": 3.6e-3, "MJ/hr": 3.6e-3,
    },
    "pressure": {
        "Pa": 1e0,
        "bar": 1e-5,
        "psi": 1e0/6894.76,
        "atm": 1e0/101325,
        "kPa": 1e-3,
        "MPa": 1e-6,
        "mmHg": 1e0/133.322,
    },
    "velocity": {
        "m/s": 1e0,
        "km/hr": 3.6,
        "mi/hr": 2.23694,
        "ft/s": 3.28084,
    },
    "angular": {
        "rad": 1e0,
        "deg": 180./np.pi,
    },
    "cost" : {
        "AUD": 1e0,
        "USD": 1.4e0,
        "MM AUD": 1e-6,
        "MM USD": 1.4e-6,
    },
    "cost_specific" : {
        "AUD/MW": 1e0,
        "USD/MW": 1.4e0,
        "MM AUD/MW": 1e-6,
        "MM USD/MW": 1.4e-6,
    },
#-------------------
    "density": {
        "kg/m3": 1e0,
        "g/cm3": 1e-3,
    },
    "specific_heat": {
        "J/kgK": 1e0, "J/kg-K": 1e0,
        "kJ/kgK": 1e-3, "kJ/kg-K": 1e-3,
    },
    "thermal_conductivity": {
        "W/mK": 1e0, "W/m-K": 1e0,
        "kW/mK": 1e-3, "kW/m-K": 1e-3,
        "J/s-m-K": 1e0, "J/s-mK": 1e0,
    },
    "viscosity": {
        "Pa-s": 1e0,
        "mPa-s": 1e3,
        "kg/m-s": 1e0
    }
}

CONVERSIONS : dict[str,dict[str|None,float]] = CONVERSIONS_DERIVED | CONVERSION_FUNDAMENTALS
UNIT_TYPES: dict[str|None, str] = dict()
for type_unit in CONVERSIONS.keys():
    for unit in CONVERSIONS[type_unit].keys():
        UNIT_TYPES[unit] = type_unit

# @dataclass(frozen=True)
@dataclass
class Variable():
    """
    Class to represent parameters and variables in the system.
    It is used to store the values with their units.
    If you have a Variable instance, you can obtain the value in different units with the u[str] method.
    In this way you make sure you are getting the value with the expected unit.
    "u" internally converts unit if it is possible.
    """

    value: float | None
    unit: str | None = None
    type: str = "scalar"

    def __add__(self, other: Self):
        """ Overloading the addition operator. """
        if not isinstance(other, Variable):
            return NotImplemented
        if self.value is None or other.value is None:
            return Variable(None, self.unit)
        if self.unit == other.unit:
            return Variable(self.value + other.value, self.unit)
        elif UNIT_TYPES[self.unit] == UNIT_TYPES[other.unit]:
            return Variable(self.value + other.u(self.unit), self.unit)
        else:
            raise TypeError(f"Cannot add {self.unit} with {other.unit}. Units are not compatible.")
        
    def __sub__(self, other: Self):
        """ Overloading the subtraction operator. """
        if not isinstance(other, Variable):
            return NotImplemented
        if self.value is None or other.value is None:
            return Variable(None, self.unit)
        if self.unit == other.unit:
            return Variable(self.value - other.value, self.unit)
        elif UNIT_TYPES[self.unit] == UNIT_TYPES[other.unit]:
            return Variable(self.value - other.u(self.unit), self.unit)
        else:
            raise TypeError(f"Cannot subtract {self.unit} with {other.unit}. Units are not compatible.")

    def __radd__(self, other: Self):
        """ Overloading the addition operator. """
        if not isinstance(other, Variable):
            return NotImplemented
        if self.value is None or other.value is None:
            return Variable(None, self.unit)
        if self.unit == other.unit:
            return Variable(self.value + other.value, other.unit)
        elif UNIT_TYPES[self.unit] == UNIT_TYPES[other.unit]:
            return Variable(other.value + self.u(other.unit), other.unit)
        else:
            raise TypeError(f"Cannot add {self.unit} with {other.unit}. Units are not compatible.")

    def __mul__(self, other: Self|float|int):
        """ Overloading the multiplication operator. """
        if isinstance(other, Variable):
            if self.value is None or other.value is None:
                return Variable(None, _mul_units(self.unit, other.unit))
            return Variable(self.value * other.value, _mul_units(self.unit, other.unit))
        elif isinstance(other, (int, float)):
            if self.value is None:
                return Variable(None, self.unit)
            return Variable(self.value * other, self.unit)
        else:
            raise TypeError(f"Cannot multiply {type(self)} with {type(other)}")

    def __rmul__(self, other: Self|float|int):
        """ Overloading the multiplication operator. """
        if isinstance(other, Variable):
            if self.value is None or other.value is None:
                return Variable(None, _mul_units(other.unit, self.unit))
            return Variable(self.value * other.value, _mul_units(other.unit, self.unit))
        elif isinstance(other, (int, float)):
            if self.value is None:
                return Variable(None, self.unit)
            return Variable(self.value * other, self.unit)
        else:
            raise TypeError(f"Cannot multiply {type(self)} with {type(other)}")

    def __truediv__(self, other: Self|float|int):
        """ Overloading the division operator. """
        if isinstance(other, Variable):
            if self.value is None or other.value is None:
                return Variable(None, _div_units(self.unit, other.unit))
            return Variable(self.value / other.value, _div_units(self.unit, other.unit))
        elif isinstance(other, (int, float)):
            if self.value is None:
                return Variable(None, self.unit)
            return Variable(self.value / other, self.unit)
        else:
            raise TypeError(f"Cannot divide {type(self)} by {type(other)}")
        
    def __eq__(self, other) -> bool:
        """ Overloading the equality operator. """
        if not isinstance(other, Variable):
            return False
        return (self.value == other.value and self.unit == other.unit)

    def __repr__(self) -> str:
        return f"{self.value:} [{self.unit}]"

    def _get_value(self, unit: str | None = None) -> float:
        """ Get the value of the variable in the requested unit.
        If the unit is not compatible with the variable unit, an error is raised.
        If the unit is None, the value is returned in the variable unit.
        """
        if unit is None:
            unit = self.unit
        if self.value is None:
            raise ValueError("Variable value is None.")
        if self.unit == unit:
            return self.value
        if UNIT_TYPES[unit] == UNIT_TYPES[self.unit]:
            if UNIT_TYPES[unit] == "temperature":
                return _conv_temp(self, unit)
            return self.value * conversion_factor(self.unit, unit)
        else:
            raise ValueError( f"Variable unit ({self.unit}) and wanted unit ({unit}) are not compatible.")

    def u(self, unit: str | None = None) -> float:
        """ Method to obtain the value of the variable in the requested unit.
        If the unit is not compatible with the variable unit, an error is raised.
        If the unit is None, the value is returned in the variable unit.
        """
        return self._get_value(unit)

    @property
    def v(self) -> float:
        """ Property to obtain the value of the variable (in its primary unit). """
        return self.value if self.value is not None else np.nan
    
    @property
    def units(self) -> str:
        """ Property to obtain the compatible units of the variable. """
        return UNIT_TYPES[self.unit]
    
    def set_unit(self, unit: str | None = None) -> None:
        """ Set the primary unit of the variable. """
        if (UNIT_TYPES[unit] == UNIT_TYPES[self.unit]) and (self.value is not None):
            self.value = self.value * conversion_factor(self.unit, unit)
            self.unit = unit
        else:
            raise ValueError(
                f"unit ({unit}) is not compatible with existing primary unit ({self.unit})."
            )
        return None

    @classmethod
    def conv_factor(cls, /, from_: str | None, final: str | None) -> Self:
        """ Class method to create a Variable instance from a value and a unit. """
        if unit is None:
            return cls(value, None)
        if unit not in CONVERSIONS:
            raise ValueError(f"Unit {unit} is not defined in the conversions.")
        return cls(value, unit)
    

#-------------------------
class Array():
    """
    Similar to Variable() but for lists (iterators, actually).
    """
    def __init__(self, values: Iterable, unit: str | None = None):
        self.values = values
        self.unit = unit
        self.type = type

    def get_values(self, unit=None) -> Iterable:
        values = self.values
        if unit is None:
            unit = self.unit

        if self.unit == unit:
            return values
        
        if UNIT_TYPES[unit] == UNIT_TYPES[self.unit]:
            conv_factor = conversion_factor(self.unit, unit)
            values_out = [v*conv_factor for v in values]
            return values_out
        if self.unit != unit:
            raise ValueError(
                f"The variable used have different units: {unit} and {self.unit}"
                )
        return values

    def __repr__(self) -> str:
        return f"{self.values:} [{self.unit}]"

def conv_factor(unit1: str|None, unit2: str|None) -> float:
    """ Function to obtain conversion factor between units.
    The units must be in the UNIT_CONV dictionary.
    If they are units from different phyisical quantities an error is raised.
    """
    if UNIT_TYPES[unit1] == "temperature" or UNIT_TYPES[unit2] == "temperature":
        raise ValueError( f"There is not conversion factor for temperature units." )
    if UNIT_TYPES[unit1] == UNIT_TYPES[unit2]:
        type_unit = UNIT_TYPES[unit1]
        conv_factor = CONVERSIONS[type_unit][unit2] / CONVERSIONS[type_unit][unit1]
    else:
        raise ValueError(f"Units {unit1=} and {unit2=} do not represent the same physical quantity.")
    return conv_factor



def conversion_factor(unit1: str|None, unit2: str|None) -> float:
    """ Function to obtain conversion factor between units.
    The units must be in the UNIT_CONV dictionary.
    If they are units from different phyisical quantities an error is raised.
    """
    if UNIT_TYPES[unit1] == "temperature" or UNIT_TYPES[unit2] == "temperature":
        raise ValueError( f"There is not conversion factor for temperature units." )
    if UNIT_TYPES[unit1] == UNIT_TYPES[unit2]:
        type_unit = UNIT_TYPES[unit1]
        conv_factor = CONVERSIONS[type_unit][unit2] / CONVERSIONS[type_unit][unit1]
    else:
        raise ValueError(f"Units {unit1=} and {unit2=} do not represent the same physical quantity.")
    return conv_factor

def _conv_temp(temp: Variable, unit: str|None) -> float:
    if temp.value is None or unit is None:
        raise ValueError("Value or unit is None")
    if temp.unit == "K" and unit == "C":
        return temp.value - 273.15
    elif temp.unit == "C" and unit == "K":
        return temp.value + 273.15
    else:
        return temp.value

def _mul_units(unit1: str|None, unit2: str|None) -> str:
    """ Function to merge two units into a single unit by multiplication.
    """
    if unit1 is None:
        return unit2 if unit2 is not None else ""
    if unit2 is None:
        return unit1
    top = []
    bottom = []
    if "/" in unit1:
        top1, bottom1 = unit1.split("/")
        top = top + top1.split("-")
        bottom = bottom + bottom1.split("-")
    else:
        top = top + unit1.split("-")
    if "/" in unit2:
        top2, bottom2 = unit2.split("/")
        top = top + top2.split("-")
        bottom = bottom + bottom2.split("-")
    else:
        top = top + unit2.split("-")
    for unit in top:
        if unit in bottom:
            top.remove(unit)
            bottom.remove(unit)
    if len(bottom) > 0:
        if len(top) == 0:
            return f"1/{'-'.join(bottom)}"
        return f"{'-'.join(top)}/{'-'.join(bottom)}"
    else:
        return f"{'-'.join(top)}"

def _div_units(unit1: str|None, unit2: str|None) -> str:
    """ Function to merge two units into a single unit by division
    """
    if unit1 is None:
        return unit2 if unit2 is not None else ""
    if unit2 is None:
        return unit1
    top = []
    bottom = []
    if "/" in unit1:
        top1, bottom1 = unit1.split("/")
        top = top + top1.split("-")
        bottom = bottom + bottom1.split("-")
    else:
        top = top + unit1.split("-")
    if "/" in unit2:
        top2, bottom2 = unit2.split("/")
        top = top + bottom2.split("-")
        bottom = bottom + top2.split("-")
    else:
        bottom = bottom + unit2.split("-")
    for unit in top:
        if unit in bottom:
            top.remove(unit)
            bottom.remove(unit)
    if len(bottom) > 0:
        if len(top) == 0:
            return f"1/{'-'.join(bottom)}"
        return f"{'-'.join(top)}/{'-'.join(bottom)}"
    else:
        return f"{'-'.join(top)}"

#---------------------
def main():

    #Examples of conversion factors and Variable usage.
    print(conversion_factor("m3/s", "L/min"))
    print(conversion_factor("W", "kJ/hr"))
    print(conversion_factor("W", "kJ/hr"))

    time_sim = Variable(365, "d")
    print(f"time_sim in days: {time_sim.u('d')}")
    print(f"time_sim in hours: {time_sim.u('hr')}")
    print(f"time_sim in seconds: {time_sim.u('s')}")

    time_sim2 = Variable(1, "d")
    print(f"time sim total: {time_sim + time_sim2}")

    time_sim2.set_unit("hr")
    print(time_sim2)
    nom_power = Variable(100, "kW")
    print(f"Energy spent: {nom_power*time_sim2}")

    nom_power.set_unit("kJ/hr")
    energy = nom_power * time_sim2
    print(f"Energy spent: {energy}")
    energy.set_unit("kWh")
    print(f"Double energy spent: {8*energy/2 - energy*2}")

    power = energy / time_sim2
    print(f"Power: {power}")

    return

if __name__=="__main__":
    main()
    pass
