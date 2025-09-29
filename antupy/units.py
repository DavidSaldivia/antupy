"""
module with a simple units manager
"""
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from antupy.core import Array
    from antupy.var import Var

BASE_UNITS: dict[str, tuple[float, str,str]] = {
    "-": (1e0, "adimensional", "adim"),
    "s": (1e0, "second", "time"),
    "m": (1e0, "meter", "length"),
    "g": (1e0, "gram", "mass"),
    "K": (1e0, "kelvin", "temperature"),
    "A": (1e0, "ampere", "current"),
    "mol": (1e0, "mole", "substance"),
    "cd": (1e0, "candela", "luminous_intensity"),
    "USD": (1e0, "us_dollar", "money")
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
    "l": (1e-3, "m3", "liter", "volume"),
    "sec": (1e0, "s", "second", "time"),
    "min": (60., "s", "minute", "time"),
    "hr": (3600., "s", "hour", "time"),
    "day": (86400., "s", "day", "time"),
    "wk": (24*3600*7, "s", "year", "time"),
    "week": (24*3600*7, "s", "year", "time"),
    "mo": (24*3600*30, "s", "year", "time"),
    "month": (24*3600*30, "s", "year", "time"),
    "yr": (31536000, "s", "year", "time"),
    "year": (31536000, "s", "year", "time"),
    "au": (149597870700, "m", "astronomic_unit", "length"),
    "mi": (1e0/1609.34,"m", "mile", "length"),
    "ft": (3.28084,"m", "foot", "length"),
    "'": (39.3701,"m", "inch", "length"),
    "ton": (1e-3,"kg", "tonne", "mass"),
    "lb": (2.20462,"kg", "pound", "mass"),
    "oz": (35.274,"kg", "ounce", "mass"),
    "lm": (1.0, "cd-sr", "lumens", "luminous_flux"),
    "Wh": (3600, "J", "watt-hour", "energy"),
    "Wp": (1e0, "W", "watt-peak", "power"),
    "cal": (4184, "J", "calorie", "energy"),
    "ha": (1e4, "m2", "hectar", "surface"),
    "°C": (1e0, "K", "celcius", "temperature"),
    "degC": (1e0, "K", "celcius", "temperature"),
    "bar": (1e5, "Pa", "bar", "pressure"),
    "psi": (6894.76, "Pa", "psi", "pressure"),
    "atm": (101325, "Pa", "atmosphere", "pressure"),
    "mmHg": (133.322, "Pa", "mm_of_mercury", "pressure"),
    "ppm": (1000, "mL/L", "parts_per_million", "concentration"),
    "deg": (np.pi/180., "rad", "degree", "plane_angle"),
    "AUD": (1.4, "USD", "AU_dollar", "money"),
    "CLP": (1e-3, "USD", "CL_pesos", "money"),
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

UnitDict = TypedDict(
    "UnitDict",
    {
        "s": int,
        "m": int,
        "g": int,
        "K": int,
        "A": int,
        "mol": int,
        "cd": int,
        "USD": int,
        "-": int,
    },
)

BASE_ADIM: UnitDict = {
    "s": 0,
    "m": 0,
    "g": 0,
    "K": 0,
    "A": 0,
    "mol": 0,
    "cd": 0,
    "USD": 0,
    "-": 0,
}

UnitPool = list[tuple[str, int]]

class Unit():
    """
    Class containing any unit valid with SI unit system.
    To initiate it, pass a "unit label", which correspond to a valid str.
    It converts it internally to a base representation, which is a dictionary
    with the 7 base SI units as keys and their respective exponents as values.

    Parameters
    ----------
    unit : str, optional
        Valid string with the unit label, e.g. "kg-m/s2". Default is "-" (dimensionless).
    base_factor : float, optional
        Multiplicative factor to convert to base SI units. Default is 1.0.

    Attributes
    ----------
    label_unit : str
        The original unit label string.
    base_factor : float
        Multiplicative factor to convert to base SI units.
    base_units : UnitDict
        Dictionary with the 7 base SI units as keys and their respective exponents as values.

    Examples
    --------
    Creating units from labels:
    
    >>> u1 = Unit("kg-m/s2")
    >>> print(u1)
    [kg-m/s2]
    >>> print(u1.si)
    1.00e+03[m-g/s2]
    
    Unit equivalence:
    
    >>> u2 = Unit("N")
    >>> print(u2)
    [N]
    >>> print(u2.si)
    1.00e+03[m-g/s2]
    >>> u1 == u2
    True
    
    Notes
    -----
    The base representation uses seven SI base units: meter (m), gram (g), second (s), 
    ampere (A), kelvin (K), mole (mol), and candela (cd). Note that gram is used instead 
    of kilogram to simplify prefix handling.
    
    See Also
    --------
    Var : Variable class that uses Unit for dimensional consistency
    Array : Array class that uses Unit for dimensional consistency
    """

    def __init__(self, unit: str = "-", base_factor: float = 1e0):
        self.base_units: UnitDict = BASE_ADIM.copy()
        self.base_factor: float = base_factor
        self.label_unit: str = unit
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
        """
        Returns the unit in base SI representation.
        The base SI representation is a string with the base factor and the base units in integer exponents
        """
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

    @property
    def u(self)->str:
        """
        Returns the unit label.
        This is just a shorter alias for label_unit."""
        return self.label_unit 

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
            if comp == "":
                name = comp
                exponent = exp_sign
            elif comp[-1].isdigit():
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
                factor = PREFIXES[name[0]] ** (exponent*exp_sign)
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

def _conv_temp(temp: Var|Array, unit: str|None) -> float|np.ndarray:
    if temp.value is None or unit is None:
        raise ValueError("Value or unit is None")
    if temp.unit.u == "K" and unit in ["°C", "degC"]:
        return temp.value - 273.15
    elif temp.unit.u in ["°C", "degC"] and unit == "K":
        return temp.value + 273.15
    elif (temp.unit.u in ["°C", "degC"] and unit in ["°C", "degC"]):
        return temp.value
    elif temp.unit.u == unit:
        return temp.value
    else:
        raise ValueError(f"either {temp.unit.u} and/or {unit} is/are incompatible.")

def _assign_unit(unit: str|Unit|None = None) -> Unit:
    if isinstance(unit, str):
        return Unit(unit)
    elif isinstance(unit, Unit):
        return unit
    else:
        raise TypeError(f"{type(unit)} is not a valid type for unit.")

def _mul_units(unit1: str|None, unit2: str|None) -> str:
    """ Function to merge two units into a single unit by multiplication.
    """
    admin_units = ["", "-", "adim"]
    if unit1 is None:
        return unit2 if unit2 is not None else ""
    if unit2 is None:
        return unit1
    if unit1 in admin_units and unit2 not in admin_units:
        return unit2
    if unit2 in admin_units and unit1 not in admin_units:
        return unit1
    if unit1 in admin_units and unit2 in admin_units:
        return "-"

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
    admin_units = ["", "-", "adim"]
    if unit1 is None:
        return unit2 if unit2 is not None else ""
    if unit2 is None:
        return unit1
    if unit2 in admin_units and unit1 not in admin_units:
        return unit1
    if unit1 in admin_units and unit2 in admin_units:
        return "-"

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
    



CONSTANTS: dict[str, tuple[float, str]] = {
    "delta_v_c": (9192631770, "Hz"), # Hyperfine transition frequency of 133Cs
    "c": (299792458, "m/s"),  # Speed of light
    "h": (6.62607015e-34, "J*s"),  # Planck's constant
    "e": (1.602176634e-19, "C"),  # Elementary charge
    "k": (1.380649e-23, "J/K"),  # Boltzmann constant
    "N_A": (6.02214076e23, "1/mol"),  # Avogadro constant
    "K_cd": (683, "lm/W"),  # Luminous efficacy of 540 THz radiation
}

USEFUL_QUANTITIES = {

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

#---------------------
def main():

    return

if __name__=="__main__":
    main()
    pass
