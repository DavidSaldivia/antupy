from typing import Literal

import numpy as np
from antupy.core.units import Unit
from antupy.core.var import Var
from antupy.utils import props, htc

import CoolProp.CoolProp as CP



_DEFAULT_FLUID = "water"
_DEFAULT_TEMP = Var(273.15, "K")
_DEFAULT_RHO = Var(999.84, "kg/m3")

_PROPERTIES_SUPERHEATED = ["temp", "rho", "p", "v", "u", "h", "s"]
_PROPERTIES_SATURATED = ["temp", "rho", "p", "q", "v", "u", "h", "s"]


_STATUS_STATE = Literal[
    "CLEAN",
    "ISO-CURVE",
    "UNDETERMINED",
    "DETERMINED",
    "DETERMINABLE",
    "OVERDETERMINED"
]

_PROPS_COOLPROP: dict[str, tuple[str, str]] = {
    "temp": ("T", "K"),
    "rho": ("D", "kg/m3"),
    "p": ("P", "Pa"),
    "q": ("Q", "-"),
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
            p: Var = Var(None, "kPa"),
            v: Var = Var(None, "m3/kg"),
            u: Var = Var(None, "kJ/kg"),
            h: Var = Var(None, "kJ/kg"),
            s: Var = Var(None, "kJ/kg-K"),
            q: Var = Var(None,"-"),
            lazy: bool = False
    ):

        self.status: _STATUS_STATE = "CLEAN"
        self.fluid_index: int = _fluid_index(fluid)
        self.fluid_label: str = fluid
        self._temp: Var = temp
        self._rho: Var = rho
        self._p: Var = p
        self._v: Var = v
        self._u: Var = u
        self._h: Var = h
        self._s: Var = s
        self._q: Var = q
        self.lazy = lazy

        if not lazy:
            self._solve_state()

    def _solve_state(self):
        props_provided: list[str] = []
        for prop in _PROPS_COOLPROP.keys():
            value_prop = Var(getattr(self, f"_{prop}"))
            value = value_prop.v if isinstance(value_prop, Var) else np.nan
            if value is not None and not np.isnan(value):
                if value_prop.unit.base_exps != Unit(_PROPS_COOLPROP[prop][1]).base_exps:
                    raise ValueError(f"Provided unit ({value_prop.u}) for property {prop} is not compatible. Required unit must be compatible with: {_PROPS_COOLPROP[prop][1]}.")
                props_provided.append(prop)
        self.props_provided = props_provided
        count_provided = len(props_provided)

        if count_provided == 0:
            self.status = "CLEAN"
            print(f"Warning: No property provided. The state is CLEAN and not solved.")
            return
        elif count_provided == 1:
            self.status = "ISO-CURVE"
            print(f"Warning: Only 1 property provided. The state is ISO-CURVE and cannot be solved. You need one property more. Provided property: {props_provided[0]}.")
            return
        elif count_provided == 2:
            self.status = "DETERMINABLE"
        elif count_provided > 2:
            self.status = "OVERDETERMINED"
            print(f"Warning: More than 2 properties provided to solve the state. Provided properties: {props_provided}. Only the first 2 will be used to solve the state.")

        if self.status in ["DETERMINABLE", "OVERDETERMINED"]:
            prop_1_label = props_provided[0]
            prop_2_label = props_provided[1]
            value_1 = Var(getattr(self, f"_{prop_1_label}")).gv(_PROPS_COOLPROP[prop_1_label][1])
            value_2 = Var(getattr(self, f"_{prop_2_label}")).gv(_PROPS_COOLPROP[prop_2_label][1])

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
                    setattr(self, f"_{prop_required}", prop_returned)
                self.status = "DETERMINED"
            except Exception as err:
                raise err

    def _check_retrievable(self, prop: str) -> None:
        if self.status == "CLEAN":
            raise ValueError("The state is not solvable. No property provided.")
        elif self.status == "ISO-CURVE":
            if prop in self.props_provided:
                return
            raise ValueError("The state is not solvable. Only one property provided.")
        elif self.status == "DETERMINED":
            return
        elif self.status == "OVERDETERMINED":
            return
        else:
            raise ValueError("The state is in an undetermined status. Something went wrong.")

    @property
    def temp(self) -> Var:
        if self.lazy:
            self._solve_state()
        self._check_retrievable("temp")
        return self._temp

    @property
    def rho(self) -> Var:
        if self.lazy:
            self._solve_state()
        self._check_retrievable("rho")
        return self._rho

    @property
    def p(self) -> Var:
        if self.lazy:
            self._solve_state()
        self._check_retrievable("p")
        return self._p
    
    @property
    def q(self) -> Var:
        if self.lazy:
            self._solve_state()
        self._check_retrievable("q")
        return self._q

    @property
    def v(self) -> Var:
        if self.lazy:
            self._solve_state()
        self._check_retrievable("v")
        return self._v

    @property
    def u(self) -> Var:
        if self.lazy:
            self._solve_state()
        self._check_retrievable("u")
        return self._u
    
    @property
    def h(self) -> Var:
        if self.lazy:
            self._solve_state()
        self._check_retrievable("h")
        return self._h
    
    @property
    def s(self) -> Var:
        if self.lazy:
            self._solve_state()
        self._check_retrievable("s")
        return self._s
    

def main():
    states = [
        FluidState(
            fluid="water",
            temp=Var(450,"degC"),
            p=Var(2, "MPa")
        ),
        FluidState(
            fluid="water",
            temp=Var(450,"degC"),
            rho=Var(999.84, "kg/m3")
        ),
        FluidState(
            fluid="water",
            p=Var(2, "MPa"),
            rho=Var(999.84, "kg/m3")
        ),
        FluidState(
            fluid="water",
            p=Var(0.1, "MPa"),
            temp=Var(300, "K")
        ),
        FluidState(
            fluid="water",
            p=Var(1, "atm"),
            q=Var(1.0, "-")
        ),
        FluidState(
            fluid="R134a",
            p=Var(200, "kPa"),
            temp=Var(-10, "degC")
        ),
    ]
    for state in states:
        print(state.temp)
        print(state.p)
        print(state.v)
        print(state.h)
        print(state.s)
        print(state.q)

if __name__ == "__main__":
    main()