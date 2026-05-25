from dataclasses import dataclass

import antupy as ap

from dev.cat.cycles.fluidstate import FluidState

def _calculate_enthalpy_isentropic(h_1: ap.Var, P_2: ap.Var) -> ap.Var:
    from scipy.optimize import fsolve
    def func(h_2s):
        s_2 = ap.props.Water().entropy(h_2s, P_2)
        return None
    hint = h_1.gv("kJ/kg")
    sol = fsolve(func, hint)[0]
    return ap.Var(sol, "kJ/kg")


@dataclass
class Turbine():
    nom_power: ap.Var = ap.Var(100, "MW")
    massflowrate: ap.Var = ap.Var(None, "kg/s")
    state_in: FluidState = FluidState(
        fluid="water",
        temp=ap.Var(450,"degC"),
        pressure=ap.Var(2, "MPa")
    )
    eta_s: ap.Var = ap.Var(0.9, "-")
    pressure_out: ap.Var = ap.Var(0.1, "bar")

    def equations(self) -> tuple[ap.Var, ap.Var]:

        eta_s = self.eta_s
        h_1 = self.state_in.enthalpy
        s_2 = ap.props.Water().entropy(self.state_in.temp, self.state_in.pressure)
        h_s = _calculate_enthalpy_isentropic(s_2, self.state_in.enthalpy)
        f1 = eta_s - (h_out - h_1)/(h_s - h_1)
        f2 = self.nom_power - self.massflowrate * (h_1 - h_out)
        return f1, f2