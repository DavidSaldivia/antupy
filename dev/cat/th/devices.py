from dataclasses import dataclass

import antupy as ap

from dev.cat.th.fluidstate import FluidState

@dataclass
class Turbine():
    nom_power: ap.Var = ap.Var(100, "MW")
    massflowrate: ap.Var = ap.Var(None, "kg/s")
    state_in: FluidState = FluidState(
        fluid="water",
        temp=ap.Var(450,"degC"),
        p=ap.Var(2, "MPa")
    )
    eta_s: ap.Var = ap.Var(0.9, "-")
    state_out: FluidState = FluidState(
        fluid="water",
        p=ap.Var(0.1, "MPa")
    )

    def equations(self) -> tuple[ap.Var, ap.Var]:
        eta_s = self.eta_s
        nom_power = self.nom_power
        massflowrate = self.massflowrate
        state_in = self.state_in
        state_out = self.state_out
        state_out_s = FluidState(
            fluid=state_out.fluid_label,
            p=state_out.p,
            s=state_in.s
        )
        f1 = eta_s - (state_out.h - state_in.h)/(state_out_s.h - state_in.h)
        f2 = nom_power - massflowrate * (state_in.h - state_out.h)
        return f1, f2
    
    def solve(self, hint: tuple[str, ap.Var] | None = None) -> None:
        if hint is None:
            raise ValueError("Hint is required to solve the turbine equations.")
        if "." in hint[0]:
            state_label, prop_label = hint[0].split(".")
        else:
            pass
            