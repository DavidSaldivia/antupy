"""
module with a simple units manager
"""
from __future__ import annotations
import numpy as np
from typing import Iterable, Self
from dataclasses import dataclass, field


BASE_UNITS: dict[str, tuple[float, str,str]] = {
    "-": (1e0, "adimensional", "adim"),
    "s": (1e0, "second", "time"),
    "m": (1e0, "meter", "length"),
    "g": (1e0, "kilogram", "mass"),
    "K": (1e0, "Kelvin", "temperature"),
    "A": (1e0, "Ampere", "current"),
    "mol": (1e0, "mole", "substance"),
    "cd": (1e0, "candela", "luminous_intensity"),
}

DERIVED_UNITS: dict[str, tuple[float,str,str,str]] = {
    "rad": (1e0, "-", "radian", "plane_angle"),
    "sr": (1e0, "-", "steradian", "solid_angle"),
    "Hz": (1e0, "1/s", "hertz", "frequency"),
    "N": (1e0, "kg-m/s2", "newton", "force"),
    "Pa": (1e0, "kg/m-s2", "pascal", "pressure"),
    "J": (1e0, "kg-m2/s2", "joule", "energy"),
    "W": (1e0, "kg-m2/s3", "watt", "power"),
    "C": (1e0, "s-A", "coulomb", "electric_charge"),
    "V": (1e0, "kg-m2/s3-A", "volt", "electric_potential"),
    "F": (1e0, "s4-A2/kg-m2", "farad", "capacitance"),
    "Ω": (1e0, "kg-m2/s3-A2", "ohm", "electrical_resistance"),
    "S": (1e0, "s3-A2/kg-m2", "siemens", "electrical_conductance"),
    "Wb": (1e0, "kg-m2/s2-A", "weber", "magnetic_flux"),
    "T": (1e0, "kg/s2-A", "tesla", "magnetic_flux_density"),
    "H": (1e0, "kg-m2/s2-A2", "henry", "inductance"),
    "lm": (1e0, "cd-sr", "lumen", "luminous flux"),
    "lx": (1e0, "cd-sr/m2", "lux", "illuminance"),
    "Bq": (1e0, "1/s", "becquerel", "radioactivity"),
    "Gy": (1e0, "m2/s2", "gray", "absorbed_dose"),
    "Sv": (1e0, "m2/s2", "sievert", "dose_equivalent"),
    "kat": (1e0, "mol/s", "katal", "catalytic_activity"),
}

RELATED_UNITS: dict[str, tuple[float,str,str,str]] = {
    "L": (1e-3, "m3", "liter", "volume"),
    "sec": (1e0, "s", "second", "time"),
    "min": (60., "s", "minute", "time"),
    "hr": (3600., "s", "hour", "time"),
    "day": (86400, "s", "day", "time"),
    "yr": (31536000, "s", "year", "time"),
    "Wh": (3600, "J", "watt-hour", "energy"),
    "cal": (4184, "J", "calorie", "energy"),
    "ha": (1e4, "m2", "hectar", "surface"),
    "°C": (1e0, "K", "celcius", "temperature"),
    "degC": (1e0, "K", "celcius", "temperature"),
}

PREFIXES: dict[str, float] = {
    "q": 1e-30, # "quecto"
    "r": 1e-27, # "ronto"
    "y": 1e-24, # "yocto"
    "z": 1e-21, # "zepto"
    "a": 1e-18, # "atto"
    "f": 1e-15, # "femto"
    "p": 1e-12, # "pico"
    "n": 1e-9, # "nano"
    "μ": 1e-6, # "micro"
    "m": 1e-3, # "milli"
    "c": 1e-2, # "centi"
    "d": 1e-1, # "deci"
    "": 1.0,
    "k": 1e3, # "kilo"
    "M": 1e6, # "mega"
    "G": 1e9, # "giga"
    "T": 1e12, # "tera"
    "P": 1e15, # "peta"
    "E": 1e18, # "exa"
    "Z": 1e21, # "zetta"
    "Y": 1e24, # "yotta"
    "R": 1e27, # "ronna"
    "Q": 1e30, # "quetta"
}

from typing import TypedDict
class UnitDict(TypedDict, total=False):
    s: int
    m: int
    g: int
    K: int
    A: int
    mol: int
    cd: int

BASE_ADIM: UnitDict = {
    "s": 0,
    "m": 0,
    "g": 0,
    "K": 0,
    "A": 0,
    "mol": 0,
    "cd": 0,
}

UnitPool = list[tuple[str, int]]

class Unit():
    """
    Class containing any unit string in its base units expression
    """

    def __init__(self, unit: str = "-", base_factor: float = 1e0):
        self.base_units: UnitDict = BASE_ADIM.copy()
        self.base_factor: float = base_factor
        self.label_unit = unit
        self._translate_to_base()

    def __repr__(self) -> str:
        return f"[{self.label_unit}]"
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Unit):
            return (
                (self.base_factor==other.base_factor)
                and (self.base_units == other.base_units)
            )
        return False
    
    @property
    def si(self) -> str:
        top_str = ""
        bottom_str = ""
        d = [(k,int(v)) for (k,v) in self.base_units.items()]    #type: ignore
        for (comp,exp) in d:
            if exp>0:
                expr = f"{comp}{abs(exp)}" if exp>1 else f"{comp}"
                if top_str == "":
                    top_str = expr
                else:
                    top_str = top_str + f"-{expr}"
            elif exp<0:
                expr = f"{comp}{abs(exp)}" if exp<-1 else f"{comp}"
                if bottom_str == "":
                    bottom_str = expr
                else:
                    bottom_str = bottom_str + f"-{expr}"
            else:
                continue
        if bottom_str == "":
            return f"{self.base_factor:.2e}[{top_str}]" if top_str != "" else "-"
        else:
            return f"{self.base_factor:.2e}[{top_str if top_str != "" else "1"}/{bottom_str}]"

    def _update_base_repr(self, name: str, exponent: int):
        exponent_prev = self.base_units.get(name,0)
        self.base_units[name] = exponent+exponent_prev
        return

    @staticmethod
    def _parse_unit_comps(
        unit_pool: UnitPool,
        comps: list[str],
        exp_sign: int
    ) -> tuple[UnitPool, float]:
        UNITS = BASE_UNITS | DERIVED_UNITS | RELATED_UNITS
        factor_ = 1.0
        for comp in comps:
            if comp[-1].isdigit():
                name = comp[:-1]
                exponent = exp_sign * int(comp[-1])
            else:
                name = comp
                exponent = exp_sign
            if name in UNITS:
                factor = 1.0
            elif name == "":
                factor = 1.0
            elif name[0] in PREFIXES and name[1:] in UNITS:
                factor = PREFIXES[name[0]] * UNITS[name[1:]][0]
                name = name[1:]
            else:
                raise ValueError(f"Unit '{name}' not recognized.")
            unit_pool.append((name, exponent))
            factor_ *= factor
        return unit_pool, factor_
                                                                   
    @classmethod
    def _split_unit(cls, unit: str) -> tuple[float, UnitPool]:
        """
        Split a unit label into its components, their factors and exponents.
        For example, "kg-m/s2" becomes [("kg", 1), ("m", 1), ("s", -2)].
        """
        unit_pool: UnitPool = []
        if unit in ["-", "", "adim"]:
            return 1.0, [("-", 0)]
        if "/" in unit:
            top, bottom = unit.split("/", 1)
            top_units = top.split("-") if "-" in top else [top,]
            bottom_units = bottom.split("-") if "-" in bottom else [bottom,]
        else:
            top_units = unit.split("-") if "-" in unit else [unit,]
            bottom_units = []
        unit_pool, factor_top = cls._parse_unit_comps(unit_pool, top_units, 1)
        unit_pool, factor_bot = cls._parse_unit_comps(unit_pool, bottom_units, -1)
        return (factor_top/factor_bot, unit_pool)

    def _translate_to_base(self) -> None:
        factor_, unit_pool_ = self._split_unit(self.label_unit)
        factor_ = self.base_factor * factor_
        while len(unit_pool_)>0:
            (name, exponent) = unit_pool_.pop(0)
            if name in BASE_UNITS:
                self._update_base_repr(name, exponent)
            if name in DERIVED_UNITS|RELATED_UNITS:
                new_label = (DERIVED_UNITS|RELATED_UNITS)[name][1]
                new_factor1 = (DERIVED_UNITS|RELATED_UNITS)[name][0]
                new_factor2, new_pool = self._split_unit(new_label)
                for comp in new_pool:
                    unit_pool_.append((comp[0], exponent*comp[1]))
                factor_ *= (new_factor2*new_factor1)**np.sign(exponent)
            self.base_factor = factor_
        return None



CONSTANTS: dict[str, tuple[float, str]] = {
    "delta_v_c": (9192631770, "Hz"), # Hyperfine transition frequency of 133Cs
    "c": (299792458, "m/s"),  # Speed of light
    "h": (6.62607015e-34, "J*s"),  # Planck's constant
    "e": (1.602176634e-19, "C"),  # Elementary charge
    "k": (1.380649e-23, "J/K"),  # Boltzmann constant
    "N_A": (6.02214076e23, "1/mol"),  # Avogadro constant
    "K_cd": (683, "lm/W"),  # Luminous efficacy of 540 THz radiation
}



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
