from dataclasses import dataclass

from antupy import Var, Array, Plant, Parametric
from antupy import props, htc

from antupy.cat.cycles.fluidstate import FluidState

def _calculate_enthalpy_isentropic(h_1: Var, P_2: Var) -> Var:
    from scipy.optimize import fsolve
    def func(h_2s):
        s_2 = props.Water().entropy(h_2s, P_2)
        return None
    hint = h_1.gv("kJ/kg")
    sol = fsolve(func, hint)[0]
    return Var(sol, "kJ/kg")


@dataclass
class Turbine():
    nom_power: Var = Var(100, "MW")
    state_in: FluidState = FluidState(
        fluid="water",
        temp=Var(450,"degC"),
        pressure=Var(2, "MPa")
    )
    eta_s: Var = Var(0.9, "-")
    pressure_out: Var = Var(0.1, "bar")

    def equations(self) -> tuple[Var, Var]:

        eta_s = self.eta_s
        h_1 = self.state_in.enthalpy
        s_2 = props.Water().entropy(self.state_in.temp, self.state_in.pressure)
        h_s = _calculate_enthalpy_isentropic(s_2, self.state_in.enthalpy)
        f1 = eta_s - (h_out - h_1)/(h_s - h_1)
        f2 = self.nom_power - self.massflowrate * (h_1 - h_out)
        return f1, f2