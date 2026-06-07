import numpy as np
from antupy import Var
from antupy import props, htc

import CoolProp.CoolProp as CP



_DEFAULT_FLUID = "water"
_DEFAULT_TEMP = Var(273.15, "K")
_DEFAULT_RHO = Var(999.84, "kg/m3")

_PROPERTIES_SUPERHEATED = ["temp", "rho", "pressure", "v", "u", "h", "s"]
_PROPERTIES_SATURATED = ["temp", "rho", "pressure", "quality", "v", "u", "h", "s"]

_POSSIBLE_STATUS = ["UNDETERMINED", "DETERMINED", "OVERDETERMINED"]


_PROPS_COOLPROP: dict[str, tuple[str, str]] = {
    "temp": ("T", "K"),
    "rho": ("D", "kg/m3"),
    "pressure": ("P", "Pa"),
    "quality": ("Q", "-"),
    "v": ("V", "m3/kg"),
    "u": ("U", "J/kg"),
    "h": ("H", "J/kg"),
    "s": ("S", "J/kg-K"),
}

_LIST_FLUIDS_COOLPROP = [
    "1-Butene", "Acetone", "Air", "Ammonia", "Argon",
    "Benzene", "CarbonDioxide", "CarbonMonoxide", "CarbonylSulfide", "CycloHexane",
    "CycloPropane", "Cyclopentane", "D4", "D5", "D6",
    "Deuterium", "Dichloroethane", "DiethylEther", "DimethylCarbonate", "DimethylEther",
    "Ethane", "Ethanol", "EthylBenzene", "Ethylene", "EthyleneOxide",
    "Fluorine", "HFE143m", "HeavyWater", "Helium", "Hydrogen",
    "HydrogenChloride", "HydrogenSulfide", "IsoButane", "IsoButene", "Isohexane",
    "Isopentane", "Krypton", "MD2M", "MD3M", "MD4M",
    "MDM", "MM", "Methane", "Methanol", "MethylLinoleate",
    "MethylLinolenate", "MethylOleate", "MethylPalmitate", "MethylStearate", "Neon",
    "Neopentane", "Nitrogen", "NitrousOxide", "Novec649", "OrthoDeuterium",
    "OrthoHydrogen", "Oxygen", "ParaDeuterium", "ParaHydrogen", "Propylene",
    "Propyne", "R11", "R113", "R114", "R115",
    "R116", "R12", "R123", "R1233zd(E)", "R1234yf",
    "R1234ze(E)", "R1234ze(Z)", "R124", "R1243zf", "R125",
    "R13", "R1336mzz(E)", "R134a", "R13I1", "R14",
    "R141b", "R142b", "R143a", "R152A", "R161",
    "R21", "R218", "R22", "R227EA", "R23",
    "R236EA", "R236FA", "R245ca", "R245fa", "R32",
    "R365MFC", "R40", "R404A", "R407C", "R41",
    "R410A", "R507A", "RC318", "SES36", "SulfurDioxide",
    "SulfurHexafluoride", "Toluene", "Water", "Xenon", "cis-2-Butene",
    "m-Xylene", "n-Butane", "n-Decane", "n-Dodecane", "n-Heptane",
    "n-Hexane", "n-Nonane", "n-Octane", "n-Pentane", "n-Propane",
    "n-Undecane", "o-Xylene", "p-Xylene", "trans-2-Butene",
]


def _fluid_index(fluid: str) -> int:
    FLUIDS = [x.lower() for x in _LIST_FLUIDS_COOLPROP]
    if fluid.lower() in FLUIDS:
        return FLUIDS.index(fluid.lower())
    else:
        raise ValueError(f"Fluid {fluid} is not in the list of fluids supported by antupy/CoolProp.")


class FluidState():
    def __init__(
            self,
            fluid: str = _DEFAULT_FLUID,
            temp: Var = Var(None, "K"),
            rho: Var = Var(None, "kg/m3"),
            pressure: Var = Var(None, "kPa"),
            v: Var = Var(None, "m3/kg"),
            u: Var = Var(None, "kJ/kg"),
            h: Var = Var(None, "kJ/kg"),
            s: Var = Var(None, "kJ/kg-K"),
            quality: Var = Var(None,"-"),
            lazy: bool = False
    ):

        self.status: str = "UNDETERMINED"
        self.fluid_index: int = _fluid_index(fluid)
        self.fluid_label: str = fluid
        self._temp: Var = temp.su("K")
        self._rho: Var = rho.su("kg/m3")
        self._pressure: Var = pressure.su("Pa")
        self._v: Var = v.su("m3/kg")
        self._u: Var = u.su("J/kg")
        self._h: Var = h.su("J/kg")
        self._s: Var = s.su("J/kg-K")
        self._quality: Var = quality.su("-")
        self.lazy = lazy

        if not lazy:
            self._solve_state()

    def _solve_state(self):
        props_provided: list[str] = []
        for prop in _PROPS_COOLPROP.keys():
            value_prop = getattr(self, f"_{prop}")
            value = value_prop.v if isinstance(value_prop, Var) else np.nan
            if value is not None and not np.isnan(value):
                props_provided.append(prop)
        count_provided = len(props_provided)

        if count_provided < 2:
            self.status = "UNDETERMINED"
            print(f"Warning: None or one property was provided. The state is UNDETERMINED and is not solved.")
            return
        
        elif count_provided == 2:
            self.status = "DETERMINED"
        elif count_provided > 2:
            self.status = "OVERDETERMINED"
            print(f"Warning: More than 2 properties provided to solve the state. Provided properties: {props_provided}. Only the first 2 will be used to solve the state.")

        if self.status in ["DETERMINED", "OVERDETERMINED"]:
            prop_1_label = props_provided[0]
            prop_2_label = props_provided[1]
            value_1 = Var(getattr(self, f"_{prop_1_label}")).v
            value_2 = Var(getattr(self, f"_{prop_2_label}")).v

            # Solving the state, if CoolProp cannot solve for any prop, it'll raise the Exception
            props_required = [p for p in _PROPS_COOLPROP.keys() if p not in props_provided]
            try:
                for prop_required in props_required:
                    prop_returned = Var(
                        CP.PropsSI(
                            _PROPS_COOLPROP[prop_required][0],
                            _PROPS_COOLPROP[prop_1_label][0],
                            value_1,
                            _PROPS_COOLPROP[prop_2_label][0],
                            value_2,
                            _LIST_FLUIDS_COOLPROP[self.fluid_index]
                        ),
                        _PROPS_COOLPROP[prop_required][1]
                    )
                    setattr(self, prop_required, prop_returned)
            except Exception as err:
                raise err

    @property
    def temp(self) -> Var:
        if self.lazy:
            self._solve_state()
        return self._temp

    @property
    def rho(self) -> Var:
        if self.lazy:
            self._solve_state()
        return self._rho
    
    @property
    def pressure(self) -> Var:
        if self.lazy:
            self._solve_state()
        return self._pressure
    
    @property
    def quality(self) -> Var:
        if self.lazy:
            self._solve_state()
        return self._quality

    @property
    def v(self) -> Var:
        if self.lazy:
            self._solve_state()
        return self._v

    @property
    def u(self) -> Var:
        if self.lazy:
            self._solve_state()
        return self._u
    
    @property
    def h(self) -> Var:
        if self.lazy:
            self._solve_state()
        return self._h
    
    @property
    def s(self) -> Var:
        if self.lazy:
            self._solve_state()
        return self._s
    

    