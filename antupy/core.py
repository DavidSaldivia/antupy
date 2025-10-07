"""
module with the core classes for AntuPy
"""
from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TypedDict

class Output(TypedDict):
    pass

@dataclass
class Simulation():
    out: dict = field(default_factory=dict)

    def __post_init__(self): ...

    def run_simulation(self, verbose: bool = True) -> None: ...


class Analyser():
    def get_simulation_instance(self, cases: Iterable) -> Simulation:
        return Simulation()
    def run_simulation(self) -> Output:
        return Output()


if __name__ == "__main__":
    import doctest
    doctest.testmod()



