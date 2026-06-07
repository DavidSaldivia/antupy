from dataclasses import dataclass

import antupy as ap

from dev.cat.th.fluidstate import FluidState

def _get_isentropic_state(state_1: FluidState, state_2: FluidState) -> FluidState:
    if state_2.pressure.v is not None:
        state_2s = FluidState(
            fluid=state_2.fluid_label,
            pressure=state_2.pressure,
            s=state_1.s
        )
    elif state_2.temp is not None:
        state_2s = FluidState(
            fluid=state_2.fluid_label,
            temp=state_2.temp,
            s=state_1.s
        )
    elif state_2.rho is not None:
        state_2s = FluidState(
            fluid=state_2.fluid_label,
            rho=state_2.rho,
            s=state_1.s
        )
    else:
        raise ValueError("---")

    return state_2s


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
    state_out: FluidState = FluidState(
        fluid="water",
        pressure=ap.Var(0.1, "MPa")
    )

    def equations(self) -> tuple[ap.Var, ap.Var]:
        eta_s = self.eta_s
        nom_power = self.nom_power
        massflowrate = self.massflowrate
        state_in = self.state_in
        state_out = self.state_out

        h_out_s = _get_isentropic_state(state_in, state_out).h
        f1 = eta_s - (state_out.h - state_in.h)/(h_out_s - state_in.h)
        f2 = nom_power - massflowrate * (state_in.h - state_out.h)
        return f1, f2
    
    def solve(self, hint: tuple[str, ap.Var] | None = None) -> None:
        if hint is None:
            raise ValueError("Hint is required to solve the turbine equations.")
        if "." in hint[0]:
            state_label, prop_label = hint[0].split(".")
        else:
            pass
            