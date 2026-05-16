from __future__ import annotations
from typing import TypeAlias, Iterable, TypedDict
from dataclasses import dataclass, field

from antupy import Var, Array, Frame

SimulationOutput: TypeAlias = dict[str, "Var | Array | Frame | float | str | dict"]

class Output(TypedDict):
    pass

@dataclass
class Simulation():
    out: SimulationOutput = field(default_factory=dict)

    def __post_init__(self): ...

    def run_simulation(self, verbose: bool = True) -> SimulationOutput: ...


class Analyser():
    def get_simulation_instance(self, cases: Iterable) -> Simulation:
        return Simulation()
    def run_simulation(self) -> Output:
        return Output()
