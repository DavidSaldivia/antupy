import numpy as np
from antupy import Var
from antupy import props, htc

import CoolProp.CoolProp as CP

_VARS = ["temp", "rho", "pressure", "quality", "v", "u", "h", "s"]
_MAPPING_VARS_COOLPROP = {
    "temp": "T",
    "rho": "D",
    "pressure": "P",
    "quality": "Q",
    "v": "V",
    "u": "U",
    "h": "H",
    "s": "S"
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

class FluidState():

    def __init__(
            self,
            fluid: str = "water",
            temp: Var = Var(273.15, "K"),
            rho: Var = Var(999.84, "kg/m3"),
            pressure: Var = Var(None, "kPa"),
            v: Var = Var(None, "m3/kg"),
            quality: Var = Var(None,"-"),
            u: Var = Var(None, "kJ/kg"),
            h: Var = Var(None, "kJ/kg"),
            s: Var = Var(None, "kJ/kg-K"),
            lazy: bool = False
    ):
        self.fluid_index: int = self._fluid_index(fluid)
        self._temp: Var = temp.su("K")
        self._rho: Var = rho.su("kg/m3")
        self._pressure: Var = pressure.su("kPa")
        self._quality: Var = quality.su("-")
        self._v: Var = v.su("m3/kg")
        self._u: Var = u.su("kJ/kg")
        self._h: Var = h.su("kJ/kg")
        self._s: Var = s.su("kJ/kg-K")

        if not lazy:
            self._solve_state()

    def _solve_state(self):
        vars_provided = []
        for var in _VARS:
            value_var = getattr(self, f"_{var}")
            value = value_var.v if isinstance(value_var, Var) else np.nan
            if value is not None and not np.isnan(value):
                vars_provided.append(var)
        
        if len(vars_provided) >= 2:
            if len(vars_provided) > 2:
                print(f"Warning: More than 2 variables provided to solve the state. Provided variables: {vars_provided}. Only the first 2 will be used to solve the state.")
            
            var_1 = vars_provided[0]
            var_2 = vars_provided[1]
            value_1 = getattr(self, f"_{var_1}").v
            value_2 = getattr(self, f"_{var_2}").v
            var_asked = [v for v in _VARS if v not in vars_provided][0]

            # check if CoolProp can solve for the asked variable:
            value_asked = CP.PropsSI(
                _MAPPING_VARS_COOLPROP[var_asked],
                var_1, value_1,
                var_2, value_2,
                _LIST_FLUIDS_COOLPROP[self.fluid_index]
            )
        elif len(vars_provided) < 2:
            raise ValueError(f"Not enough variables provided to solve the state. Provided variables: {vars_provided}")

        pass

    def _fluid_index(self, fluid: str) -> int:
        FLUIDS = [x.lower() for x in _LIST_FLUIDS_COOLPROP]
        if fluid.lower() in FLUIDS:
            return FLUIDS.index(fluid.lower())
        else:
            raise ValueError(f"Fluid {fluid} is not in the list of fluids supported by antupy/CoolProp.")


    @property
    def temp(self) -> Var:
        return self._temp

    @property
    def rho(self) -> Var:
        return self._rho
    
    @property
    def pressure(self) -> Var:
        return self._pressure
    
    @property
    def quality(self) -> Var:
        return self._quality

    @property
    def v(self) -> Var:
        return self._v

    @property
    def u(self) -> Var:
        return self._u
    
    @property
    def h(self) -> Var:
        return self._h
    
    @property
    def s(self) -> Var:
        return self._s
    

    @property
    def determined(self) -> bool:
        return self._temp is not None and self._rho is not None

    